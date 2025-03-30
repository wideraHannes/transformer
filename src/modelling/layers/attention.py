from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Unified scaled dot-product attention supporting both single-head and multi-head formats."""

    def __init__(self, mask_future: bool = False):
        super().__init__()
        self.mask_future = mask_future
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute scaled dot-product attention.

        Automatically handles both single-head (3D) and multi-head (4D) inputs:
        - Single-head: (batch_size, seq_length, dim)
        - Multi-head: (batch_size, num_heads, seq_length, dim)

        Args:
            q: Query tensor of shape (..., seq_length_q, dim)
            k: Key tensor of shape (..., seq_length_k, dim)
            v: Value tensor of shape (..., seq_length_k, dim)
            mask: Optional mask tensor, broadcastable to (..., seq_length_q, seq_length_k)

        Returns:
            Attention output with same shape as query
        """
        # Get scaling factor from key dimension
        d_k = k.size(-1)
        scale = torch.sqrt(torch.tensor(d_k, dtype=torch.float32, device=k.device))

        # Compute scaled attention scores using einsum
        # This works for both 3D and 4D inputs due to einsum's flexibility
        scores = torch.einsum("...qd,...kd->...qk", q, k) / scale

        # Apply future mask if enabled
        if self.mask_future:
            future_mask = torch.triu(
                torch.ones(scores.size(-2), scores.size(-1), device=scores.device),
                diagonal=1,
            ).bool()
            # Add necessary dimensions for broadcasting
            future_mask = future_mask.expand(scores.shape)
            scores = scores.masked_fill(future_mask, float("-inf"))

        # Apply attention mask if provided
        if mask is not None:
            # Expand mask for broadcasting if needed
            while mask.dim() < scores.dim():
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Compute attention weights and final output
        attn_weights = F.softmax(scores, dim=-1)
        # attn_weights = self.dropout(attn_weights)
        return torch.einsum("...qk,...kd->...qd", attn_weights, v)
