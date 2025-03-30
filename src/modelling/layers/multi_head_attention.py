from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modelling.layers.attention import Attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, mask_future=False):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.query_transform = nn.Linear(d_model, d_model, bias=False)
        self.key_transform = nn.Linear(d_model, d_model, bias=False)
        self.value_transform = nn.Linear(d_model, d_model, bias=False)
        self.output_transform = nn.Linear(d_model, d_model, bias=False)

        self.mask_future = mask_future
        self.attention = Attention(mask_future=mask_future)

        # Initialize weights
        nn.init.xavier_uniform_(self.query_transform.weight)
        nn.init.xavier_uniform_(self.key_transform.weight)
        nn.init.xavier_uniform_(self.value_transform.weight)
        nn.init.xavier_uniform_(self.output_transform.weight)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        query = (
            self.query_transform(query)
            .view(batch_size, -1, self.n_heads, self.d_k)
            .transpose(1, 2)
        )  # Shape: (batch_size, n_heads, seq_len, d_k)
        key = (
            self.key_transform(key)
            .view(batch_size, -1, self.n_heads, self.d_k)
            .transpose(1, 2)
        )  # Shape: (batch_size, n_heads, seq_len, d_k)
        value = (
            self.value_transform(value)
            .view(batch_size, -1, self.n_heads, self.d_k)
            .transpose(1, 2)
        )  # Shape: (batch_size, n_heads, seq_len, d_k)

        # Apply attention on all the projected vectors in batch
        attention_output = self.attention(query, key, value, mask)

        # Concatenate heads and put through final linear layer
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.d_model)
        )
        return self.output_transform(attention_output)


if __name__ == "__main__":
    # Define test data for hidden states and attention masks
    VALUE = torch.tensor(
        [
            [[0.0349, 0.3211, 1.5736, -0.8455], [0.0000, 0.0000, 0.0000, 0.0000]],
            [
                [-1.4181, 0.8963, 0.0499, 2.2667],
                [1.1790, -0.4345, -1.3864, -1.2862],
            ],
        ]
    )

    # Batch x Seq_length x dimension

    QUERY = torch.tensor(
        [
            [
                [1.9269, 1.4873, 0.9007, -2.1055],
                [0.6784, -1.2345, -0.0431, -1.6047],
                [0.3559, -0.6866, -0.4934, 0.2415],
            ],
            [
                [-1.1109, 0.0915, -2.3169, -0.2168],
                [-0.3097, -0.3957, 0.8034, -0.6216],
                [0.0000, 0.0000, 0.0000, 0.0000],
            ],
        ]
    )

    QUERY_ATTENTION_MASK = torch.tensor([[1, 1, 1], [1, 1, 0]])

    mha = MultiHeadAttention(QUERY.size(-1), 2, mask_future=False)
    output = mha(QUERY, VALUE, VALUE, QUERY_ATTENTION_MASK)
