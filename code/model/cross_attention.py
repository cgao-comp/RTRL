import math

import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, n_heads, dropout=0.1, data_type=torch.float32):
        super(CrossAttention, self).__init__()

        self.q_linear = nn.Linear(input_dim, hidden_dim, bias=False, dtype=data_type)
        self.k_linear = nn.Linear(input_dim, hidden_dim, bias=False, dtype=data_type)
        self.v_linear = nn.Linear(input_dim, hidden_dim, bias=False, dtype=data_type)
        self.out_linear = nn.Linear(hidden_dim, out_dim, bias=False, dtype=data_type)

        self.scale = 1. / math.sqrt(hidden_dim)

        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape

        target_embedding = self.q_linear(target_embedding).view(B, L, self.n_heads, -1)
        source_embedding = self.k_linear(source_embedding).view(S, self.n_heads, -1)
        value_embedding = self.v_linear(value_embedding).view(S, self.n_heads, -1)

        """
        target_embedding: [24, 64, 8, 128]
        source_embedding: [1000, 8, 128]
        value_embedding : [1000, 8, 128]

        out             : [24, 64, 8, 128]
        """
        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)
        A = self.dropout(torch.softmax(self.scale * scores, dim=-1))
        out = torch.einsum("bhls,she->blhe", A, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_linear(out)

if __name__ == "__main__":
    torch.manual_seed(42)

    print("=== basic usage ===")
    batch_size = 128
    seq_len = 8
    input_dim = 128
    word_size = 1000
    num_heads = 8

    cross_attn = CrossAttention(
        input_dim=input_dim,
        hidden_dim=256,
        out_dim=input_dim,
        n_heads=8,
        dropout=0.1
    )

    query = torch.randn(batch_size, 2, input_dim)
    key = torch.randn(word_size, input_dim)
    value = torch.randn(word_size, input_dim)

    print(f"query shape: {query.shape}")
    print(f"key shape: {key.shape}")
    print(f"value shape: {value.shape}")

    output = cross_attn(query, key, value)
    print(f"output shape: {output.shape}")
