import torch.nn as nn
from src.modelling.layers.multi_head_attention import MultiHeadAttention
from src.modelling.layers.feed_forward import PositionwiseFeedForward


class BaseTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, feature_dim, dropout=0.1):
        super(BaseTransformerLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.feature_transformation = PositionwiseFeedForward(
            d_model, feature_dim, dropout
        )
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # Self-attention layer
        attention_output = self.self_attention(src, src, src, src_mask)
        src = src + self.dropout(attention_output)
        src = self.layer_norm_1(src)

        # Feed-forward layer
        ff_output = self.feature_transformation(src)
        src = src + self.dropout(ff_output)
        src = self.layer_norm_2(src)

        return src
