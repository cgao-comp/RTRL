import math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class Qwen3RMSNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        return (hidden_states * self.weight).to(input_dtype)


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(
            self,
            dim: int,
            max_position_embeddings: int = 2048,
            base: int = 100000,
            device: Optional[str] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = x.shape[1]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim: int = 1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3AttentionCore(nn.Module):
    def __init__(
            self,
            hidden_size: int = 1024,
            num_heads: int = 16,
            num_kv_heads: int = 8,
            head_dim: int = 128,
            max_position: int = 2048,
            rope_theta: float = 100000.0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        self.q_norm = Qwen3RMSNorm(head_dim, eps=1e-6)
        self.k_norm = Qwen3RMSNorm(head_dim, eps=1e-6)

        self.rotary_emb = Qwen3RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=max_position,
            base=rope_theta,
        )

    def copy_weight(self, ori_layer):
        with torch.no_grad():
            self.q_proj.weight.copy_(ori_layer.self_attn.q_proj.weight)
            self.k_proj.weight.copy_(ori_layer.self_attn.k_proj.weight)
            self.v_proj.weight.copy_(ori_layer.self_attn.v_proj.weight)
            self.o_proj.weight.copy_(ori_layer.self_attn.o_proj.weight)
            self.q_norm.weight.copy_(ori_layer.self_attn.q_norm.weight)
            self.k_norm.weight.copy_(ori_layer.self_attn.k_norm.weight)

    def forward(self, hidden_states: torch.Tensor):
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q_flat = q.reshape(-1, seq_len, self.head_dim)
        q = self.q_norm(q_flat).view(q.shape)

        k_flat = k.reshape(-1, seq_len, self.head_dim)
        k = self.k_norm(k_flat).view(k.shape)

        kv_seq_len = seq_len

        cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
        q, k = apply_rotary_pos_emb(
            q.transpose(1, 2), k.transpose(1, 2), cos, sin
        )
        q, k = q.transpose(1, 2), k.transpose(1, 2)

        attn = torch.matmul(q, k.repeat([1, self.num_kv_groups, 1, 1]).transpose(-2, -1)) / math.sqrt(self.head_dim)

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v.repeat([1, self.num_kv_groups, 1, 1]))

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.o_proj(out)

        return out

    @torch.no_grad()
    def infer(self, hidden_states: torch.Tensor):
        return self.forward(hidden_states)

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    model_name = "Qwen3-0.6B"
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    local_model_dir = os.path.join(_ROOT, 'saved_checkpoints', 'LLM', 'Qwen3-0_6B')
    tokenizer = AutoTokenizer.from_pretrained(
        local_model_dir,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        local_model_dir,
        trust_remote_code=True,
        config=AutoConfig.from_pretrained(os.path.join(local_model_dir, "config.json")),
        attn_implementation="eager"
    )

    print("=" * 70)
    print("testing Qwen3Attention")
    print("=" * 70)

    batch_size = 256
    seq_len = 60
    hidden_size = 1024

    test_input = torch.randn(batch_size, seq_len, hidden_size)

    print("\n" + "-" * 50)
    print("forward pass test:")
    print("-" * 50)

    attn_simple = Qwen3AttentionCore(
        hidden_size=1024,
        num_heads=16,
        num_kv_heads=8,
        head_dim=128,
    )
    attn_simple.copy_weight(model.model.layers[0])

    out_simple = attn_simple(test_input)

    print(test_input.shape)
    print(out_simple.shape)

    print("\n" + "=" * 70)
    print("all tests passed")
    print("=" * 70)
