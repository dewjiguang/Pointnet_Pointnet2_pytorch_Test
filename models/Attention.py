import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        # x shape: (batch_size, in_channels, num_points)
        query = self.conv1(x)
        key = self.conv2(x)

        # Compute attention scores
        attn_scores = torch.bmm(query.transpose(1,2), key)

        # Normalize attention scores using softmax
        attn_scores = self.softmax(attn_scores)

        # Apply attention to the input features
        attended_features = torch.bmm(x, attn_scores.transpose(1,2))

        # Apply a residual connection
        attended_features = attended_features + x

        # Apply a nonlinearity
        attended_features = self.relu(attended_features)

        return attended_features
