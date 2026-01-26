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
