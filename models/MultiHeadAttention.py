import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        # 8,512
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.d_model = d_model

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size = x.size(0)

        # Step 1: Perform linear transformation and split into multiple heads
        query = self.query_linear(x).view(batch_size, -1, self.num_heads, self.d_model // self.num_heads).transpose(1,2)
        key = self.key_linear(x).view(batch_size, -1, self.num_heads, self.d_model // self.num_heads).transpose(1,2)
        value = self.value_linear(x).view(batch_size, -1, self.num_heads, self.d_model // self.num_heads).transpose(1,2)

        # Step 2: Calculate attention scores
        scores = torch.matmul(query, key.transpose(-2,-1)) / (self.d_model // self.num_heads)**0.5

        # Step 3: Apply masking (if required)

        # Step 4: Calculate attention weights
        attn_weights = nn.functional.softmax(scores, dim=-1)

        # Step 5: Apply dropout (if required)

        # Step 6: Calculate weighted sum
        weighted_sum = torch.matmul(attn_weights, value)

        # Step 7: Concatenate and apply linear transformation
        weighted_sum = weighted_sum.transpose(1,2).contiguous().view(batch_size, -1, self.num_heads * (self.d_model // self.num_heads))
        output = self.output_linear(weighted_sum)

        return output
