import math
from transformers import PretrainedConfig

'''
configuration of model, huggingface 的一个类

'''
class CoconutMindConfig(PretrainedConfig):
    model_type = "coconutmind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        
        ############ MoE ############
        use_moe:bool=False,
        num_experts_per_tok:int=2,
        n_routed_experts:int=4,
        n_shared_experts:int=1,
        scoring_func:str='softmax',
        aux_loss_alpha:float=0.1,
        seq_aux:bool=True,
        norm_topk_prob:bool=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe=use_moe
        self.num_experts_per_tok=num_experts_per_tok
        self.n_routed_experts=n_routed_experts
        self.n_shared_experts=n_shared_experts
        self.seq_aux=seq_aux
        self.norm_topk_prob=norm_topk_prob
        self.aux_loss_alpha=aux_loss_alpha
        self.scoring_func=scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 4,
                "beta_slow": 1,
                "factor": 4,
                "original_max_position_embeddings": 2048,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )


import torch
import torch.nn as nn

# Basic structure

# hesitate nn.Module class
class RMSNorm(nn.Module):
# __init__ initialization
    def __init__(self, dim: int, eps: float=1e-5): # two parameters
        super().__init__() # we need to initialize the parent class first
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) # weight -> gammar, a learnable parameter

    '''
    假设输入 x 的 shape 是 (batch, seq_len, dim)
    x.pow(2).mean(-1, keepdim=True)，在最后一个维度上做均方，并保留最后一个维度，即 (batch, seq_len, 1)
    add(self.eps) 和 rsqrt() 都不会改变 shape

    在 forward 中，x * self._norm(x) 利用了 PyTorch 的广播机制，把 (batch, seq_len, 1) 自动扩展为 (batch, seq_len, dim)，实现每个 token 的所有 feature 都除以同一个归一化因子。
    '''

    # _norm
    def _norm(self, x: torch.Tensor):
        # 均方和 + eps 求倒数
        # an util function
        return x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()

    # forward
    def forward(self, x: torch.Tensor):
        return x * self._norm(x) * self.weight


# 实现 RoPE 编码  
def precompute_freqs_cis(
    dim: int,
    end: int,
    rope_theta: int = 1000000, # base，最大频率
    rope_scaling: dict = None,
):
    # 1. 写出 RoPE 的数学公式
    # 2. 计算 corr_dim
    # 3. 计算 beta
    # 4. 计算 scale
    # 5. 应用 scale
    # 6. 返回一个 cos 和 sin

    # 1. 写出 RoPE 的数学公式
    freqs = 1.0 / (
        rope_theta ** (torch.arange(0, dim, 2).float() / dim)
    )  # (dim/2,) # 计算每个维度对应的频率

    if rope_scaling is not None:
        # 取出 rope_scaling 的参数
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling.get("original_max_position_embeddings", 2048), # 没有就给默认值2048
            rope_scaling.get("factor", 4),
            rope_scaling.get("beta_fast", 4),
            rope_scaling.get("beta_slow", 1),
        )
        # 2. 计算 corr_dim
        corr_dim = next((i for i in range(dim//2) if 2 * math.pi/freqs[i] > orig_max), dim//2)
        # 上述代码的含义是，如果没有超过 orig_max, 就是 pure rope，否则需要插值压缩 

        # 3. 计算 power： for rope scaling
        power = torch.arange(0, dim//2, device=freqs.device).float() / (max(dim//2 -1, 1))

        # 4. 计算 beta
        beta = beta_slow + (beta_fast - beta_slow) * power

        # 5. 计算 scale
        scale = torch.where(
            torch.arange(0, dim//2, device=freqs.device) < corr_dim,
            (beta * factor - beta + 1) / (beta * factor), # 高频插值
            1.0/factor,  # 低频压缩
        )

        # 6. 应用 scale
        freqs = freqs * scale

    # 7. 生成位置索引，并计算 freqs * pos，找到对应位置的角度
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()  # 外积计算

    # 8. 返回一个 cos 和 sin
    # 需要扩一倍 -> 因为 cos 和 sin 交替出现
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)  # (end, dim)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)  # (end, dim)

    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim = 1):
    
    # 复数在实数域上的旋转
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    # x_roated = x * cos + rotate_half(x) * sin
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))

    return q_embed, k_embed