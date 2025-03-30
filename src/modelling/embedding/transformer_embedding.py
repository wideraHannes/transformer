import torch.nn as nn

from src.modelling.embedding.positional_encoding import PositionalEncoding
from src.modelling.embedding.word_embedding import WordEmbedding


class TransformerEmbedding(nn.Module):
    """
    token embedding + positional Embedding
    """

    def __init__(self, vocab_size, d_model, max_len):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len)

    def forward(self, token):
        device = token.device
        token_emb = self.token_emb(token).to(device)
        pos_emb = self.pos_emb(token).to(device)

        return token_emb + pos_emb
