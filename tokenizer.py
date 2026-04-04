# TODO: Batch encoding 

class Tokenizer:
    def __init__(self):
        standard_toks = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-']
        prepend = ["<cls>", "<pad>", "<eos>", "<unk>"]
        append  = ["<mask>"]
        # Align to a multiple of 8 with the null token for better GPU performance
        self.all_toks = prepend + standard_toks + ["<null_1>"] + append

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

        self.cls_idx = self.tok_to_idx["<cls>"]
        self.eos_idx = self.tok_to_idx["<eos>"]
        self.pad_idx = self.tok_to_idx["<pad>"]
        self.unk_idx = self.tok_to_idx["<unk>"]
        self.mask_idx = self.tok_to_idx["<mask>"]

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def encode(self, sequence):
        return [self.cls_idx] + [self.get_idx(tok) for tok in sequence] + [self.eos_idx]

    def decode(self, indices):
        return [self.get_tok(tok) for tok in indices]