import torch.nn as nn
from src.modelling.layers.multi_head_attention import MultiHeadAttention
from src.modelling.layers.feed_forward import PositionwiseFeedForward


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, feature_dim, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, mask_future=True)
        self.encoder_attention = MultiHeadAttention(d_model, n_heads)
        self.feature_transformation = PositionwiseFeedForward(
            d_model, feature_dim, dropout
        )

        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.layer_norm_3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src_emb, tgt, memory_mask=None, tgt_mask=None):
        # Self-attention layer
        self_attention_output = self.self_attention(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout(self_attention_output)
        tgt = self.layer_norm_1(tgt)

        # Cross-attention layer
        cross_attention_output = self.encoder_attention(
            tgt, src_emb, src_emb, memory_mask
        )
        tgt = tgt + self.dropout(cross_attention_output)
        tgt = self.layer_norm_2(tgt)

        # Feed-forward layer
        ff_output = self.feature_transformation(tgt)
        tgt = tgt + self.dropout(ff_output)
        tgt = self.layer_norm_3(tgt)

        return tgt
