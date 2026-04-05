"""Standalone tokenizer for ESM-2.

Provides the same vocabulary and encoding as the official ESM-2 alphabet
without requiring PyTorch or the ``facebookresearch/esm`` package.
"""

import mlx.core as mx


# Standard amino-acid tokens in ESM-2 canonical order.
STANDARD_TOKS = list("LAGVSERTIDPKQNFYMHWCXBUZO.-")

# Full vocabulary: special tokens + amino acids + null padding + mask.
VOCAB = ["<cls>", "<pad>", "<eos>", "<unk>"] + STANDARD_TOKS + ["<null_1>", "<mask>"]


class Tokenizer:
    """ESM-2 tokenizer.

    Encodes protein sequences into integer token arrays compatible with
    :class:`model.ESM2`.  Handles single sequences and batches with automatic
    padding.

    Example::

        tok = Tokenizer()
        tokens = tok.encode("MKTAYIAK")       # single sequence -> mx.array
        tokens = tok.batch_encode(["MKTAYIAK", "KALTARQ"])  # batch -> mx.array
    """

    def __init__(self):
        self.vocab = VOCAB
        self.tok_to_idx: dict[str, int] = {t: i for i, t in enumerate(self.vocab)}

        self.cls_idx: int = self.tok_to_idx["<cls>"]
        self.pad_idx: int = self.tok_to_idx["<pad>"]
        self.eos_idx: int = self.tok_to_idx["<eos>"]
        self.unk_idx: int = self.tok_to_idx["<unk>"]
        self.mask_idx: int = self.tok_to_idx["<mask>"]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _encode_one(self, sequence: str) -> list[int]:
        """Encode a single amino-acid string to a list of token indices."""
        return (
            [self.cls_idx]
            + [self.tok_to_idx.get(c, self.unk_idx) for c in sequence]
            + [self.eos_idx]
        )

    def encode(self, sequence: str) -> mx.array:
        """Encode a single sequence, returning an ``mx.array`` of shape ``(1, L+2)``."""
        return mx.array([self._encode_one(sequence)])

    def batch_encode(self, sequences: list[str]) -> mx.array:
        """Encode multiple sequences with right-padding.

        Returns an ``mx.array`` of shape ``(batch, max_len + 2)`` where
        shorter sequences are padded with ``<pad>`` tokens.
        """
        encoded = [self._encode_one(seq) for seq in sequences]
        max_len = max(len(e) for e in encoded)
        padded = [e + [self.pad_idx] * (max_len - len(e)) for e in encoded]
        return mx.array(padded)

    def decode(self, indices: mx.array | list[int]) -> list[str]:
        """Decode token indices back to token strings."""
        if isinstance(indices, mx.array):
            indices = indices.tolist()
        return [self.vocab[i] for i in indices]
