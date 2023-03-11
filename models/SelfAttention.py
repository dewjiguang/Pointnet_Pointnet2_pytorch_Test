import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, in_dim, n_heads):
        super(SelfAttention, self).__init__()

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=in_dim,
            num_heads=n_heads,
        )

        self.layer_norm = nn.LayerNorm(in_dim)

    def forward(self, x):
        # x: [batch_size, seq_len, in_dim]
        residual = x
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, in_dim]
        x, _ = self.multihead_attn(x, x, x)
        x = x.permute(1, 0, 2)  # [batch_size, seq_len, in_dim]
        x = self.layer_norm(residual + x)
        return x
