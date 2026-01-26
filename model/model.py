import math
from typing import Optional, Tuple, Union
from transformers import PretrainedConfig
from typing import Optional
from torch.nn import functional as F # functional API
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin # GenerationMixin 用于生成任务
from transformers.modeling_outputs import CausalLMOutputWithPast

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

# 首先定义一个 repeat_kv 函数，用于复制 key 和 value 张量
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # 获取 x 的形状
    bsz, seq_len, n_heads, head_dim = x.shape
    # 如果 n_rep 是 1，直接返回 x
    if n_rep == 1:
        return x
    # 为什么不直接用 x.copy() 呢？因为我们需要在 head 维度上进行复制
    return (
        x[:, :, :, None, :]  # 在 head 维度后面增加一个维度
        .expand(bsz, seq_len, n_heads, n_rep, head_dim)  # 复制 n_rep 次
        .reshape(bsz, seq_len, n_heads * n_rep, head_dim)  # 重塑形状
        # 用 reshape，不用关心内存连续性问题，因为 PyTorch 会自动处理
    )

class Attention(nn.Module):
    def __init__(self, args: CoconutMindConfig):
        super().__init__()
        # 1. 头数和每头维度
        self.n_local_heads = args.num_attention_heads  # 总头数
        self.num_key_value_heads = args.num_key_value_heads if args.num_key_value_heads is not None else args.num_attention_heads  # KV头数
        self.head_dim = args.hidden_size // args.num_attention_heads  # 每头维度
        self.n_rep = self.n_local_heads // self.num_key_value_heads  # 重复次数

        assert args.hidden_size % args.num_attention_heads == 0, "hidden size must be divisible by num attention heads"

        # 2. Q/K/V 投影层
        # 输入: (batch, seq_len, hidden_size)
        # 输出: (batch, seq_len, n_local_heads * head_dim)
        self.q_proj = nn.Linear(args.hidden_size, self.n_local_heads * self.head_dim, bias=False)
        # 输出: (batch, seq_len, num_key_value_heads * head_dim)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)

        # 3. 输出投影
        # 输入: (batch, seq_len, n_local_heads * head_dim)
        # 输出: (batch, seq_len, hidden_size)
        self.o_proj = nn.Linear(self.n_local_heads * self.head_dim, args.hidden_size, bias=False)

        # 4. Dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # 5. flash attention 支持
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention") and args.flash_attention


    # 实现 forward 函数
    # 1. 投影，计算 q/k/v
    # 2. 对 q/k 应用 RoPE 编码 
    # 3. 对 k 和 v 做重复，同时注意缓存已经计算过的 key/value
    # 4. 计算注意力
    # 5. 拼接多头输出，线性变换，dropout
    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        post_kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_chache=False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. 投影，计算 q/k/v
        bsz, seq_len, _ = x.size() # 获取输入形状
        xq = self.q_proj(x)  # (bsz, seq_len, n_local_heads * head_dim)
        xk = self.k_proj(x)  # (bsz, seq_len, num_key_value_heads * head_dim)
        xv = self.v_proj(x)  # (bsz, seq_len, num_key_value_heads * head_dim)
        # 用 view 拆分为多个头
        q = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)  # (bsz, seq_len, n_local_heads, head_dim)
        k = xk.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)  # (bsz, seq_len, num_key_value_heads, head_dim)
        v = xv.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)  # (bsz, seq_len, num_key_value_heads, head_dim)
        # 2. 对 q/k 应用 RoPE 编码
        cos, sin = position_embeddings  # (seq_len, dim), (seq_len, dim)
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])
        # 3. 对 k 和 v 做重复，同时注意缓存已经计算过的 key/value
        if not post_kv_cache is None:
            # 有缓存，说明是生成阶段
            past_k, past_v = post_kv_cache  # (bsz, past_seq_len, num_key_value_heads, head_dim)
            k = torch.cat([past_k, xk], dim=1)  # 在 seq_len 维度拼接
            v = torch.cat([past_v, xv], dim=1)
        # 更新缓存
        if use_chache:
            post_kv_cache = (k, v)
        # 重复 k 和 v, 需要交换一下 head 维度和 seq_len 维度
        # 这样 repeat_kv 函数才能在 head 维度上进行复制
        # 交换前： (bsz, seq_len, num_key_value_heads, head_dim)
        # 交换后： (bsz, num_key_value_heads, seq_len, head_dim)
        xq.transpose_(1, 2)  # (bsz, n_local_heads, seq_len, head_dim)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)  # (bsz, n_local_heads, seq_len, head_dim)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        # 计算注意力
        # 是否使用 flash attention：条件是 flash attention 可用，且序列长度大于 1，且没有 attention mask 或者 attention mask 全是 1
        if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            # 计算 flash attention
            # 计算逻辑是：
            # 如果 attention_mask 是 None，就传 None
            # 否则，调整 attention_mask 的形状为 (bsz, 1, 1, seq_len)，并扩展为 (bsz, n_local_heads, seq_len, seq_len) -> 即 在 head 维度上扩展
            attn_mask = (
                None
                if attention_mask is None
                else attention_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_local_heads, seq_len, -1).bool() 
            )
            # 计算
            out = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=True) # test 的时候不 dropout; is_causal=True 因为是自回归，即每次只能看到前面的token
        else:
            # transpose 后的形状：(bsz, n_local_heads, seq_len, head_dim) -> 在后面两个维度上计算，需要先转置
            # scores 的 形状：(bsz, n_local_heads, seq_len, seq_len)
            scores = (xq@xk.transpose(-2,-1))/math.sqrt(self.head_dim)  # (bsz, n_local_heads, seq_len, seq_len)
            # 这一行手动将上三角部分设为 -inf，实现 causal masking
            # 即所有未来的位置都不能被 attention 到
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1
            ).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
            # 为什么需要 unsqueeze 呢？因为 torch.triu 返回的形状是 (seq_len, seq_len)，需要扩展为 (1, 1, seq_len, seq_len) 才能和 scores 相加
            # 如果有 attention mask，就应用
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # (bsz, 1, 1, seq_len)
                # attention mask 的形状是 (bsz, seq_len)，需要扩展为 (bsz, 1, 1, seq_len) 才能和 scores 相加
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9  # 把 0 变成 -1e9，1 变成 0
                scores = scores + extended_attention_mask  # (bsz, n_local_heads, seq_len, seq_len)
            
            # 经过 softmax 得到注意力权重
            scores = torch.softmax(scores, dim=-1).type_as(xq)  # (bsz, n_local_heads, seq_len, seq_len)
            # dropout
            scores = self.attn_dropout(scores)
            # 计算加权和
            output = scores @ xv  # (bsz, n_local_heads, seq_len, head_dim)
        
        # 恢复维度： 和输入一致
        # (bsz, seq_len, n_local_heads * head_dim)
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        # 残差
        output = self.resid_dropout(self.o_proj(output))
        return output, post_kv_cache  # (bsz, seq_len, hidden_size)

# 定义 FFN 层
class FeedForward(nn.Module):
    # 1. 初始化
    # 2. 升维
    # 3. 降维
    # 4. 门控
    # 5. droupout
    # 6. 激活函数
    def __init__(self, args: CoconutMindConfig):
        super().__init__()
        if args.intermediate_size is None:
            intermiediate_size = int(args.hidden_size * 8 / 3) # this is a empirical value, in llama2 paper
            args.intermediate_size = 64*((intermiediate_size + 64 - 1)//64) # make it divisible by 64
        
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)  # 升维
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)  # 降维
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)  # 门控
        self.dropout = nn.Dropout(args.dropout)
        self.act_fn = ACT2FN[args.hidden_act]  # 一些常用的激活函数，比如 relu, gelu, silu 等
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 按照数据流图计算
        # 1. 先计算 up_proj(x)
        # 2. 经过激活函数之后，和 gate_proj(x) 相乘
        # 3. 经过 down_proj
        # 4. 最后经过 dropout
        return self.dropout(self.down_proj(self.act_fn(self.up_proj(x)) * self.gate_proj(x)))

# 定义 block
class CoconutMindBlock(nn.Module):
    def __init__(self, layer_id:int, args: CoconutMindConfig):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads # 注意力头数
        self.hidden_size = args.hidden_size # 隐藏层大小
        self.head_dim = args.hidden_size // args.num_attention_heads # 每个头的维度
        self.attention = Attention(args) # 注意力层

        self.layer_id = layer_id
        # 初始化输入层归一化和注意力后归一化，前馈网络
        self.input_layernorm = RMSNorm(self.hidden_size, eps=args.rms_norm_eps) # 输入层归一化
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=args.rms_norm_eps) # 注意力后归一化
        self.ffn = FeedForward(args) # 前馈网络

    def forward(
        self,
        hidden_states, # hidden states 指的是输入的 x
        position_embeddings: Tuple[torch.Tensor, torch.Tensor], # precomputed RoPE embeddings
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # cached key value 
        use_cache=False, # 默认不使用缓存
        attention_mask: Optional[torch.Tensor] = None, # attention mask
    ):
        # 1. 残差存储
        residual = hidden_states
        # 2. 输入层归一化
        # hidden_states = self.input_layernorm(hidden_states)
        # 3. 注意力
        attn_output, present_key_value = self.attention(
            hidden_states,
            position_embeddings,
            post_kv_cache=past_key_value,
            use_chache=use_cache,
            attention_mask=attention_mask,
        )
        # 4. attention 后残差连接
        hidden_states = residual + attn_output
        # 5. 前馈网络前归一化 然后残差连接
        hidden_states = hidden_states + self.ffn(self.post_attention_layernorm(hidden_states))

        return hidden_states, present_key_value

# 定义整个模型
class CoconutMindModel(nn.Module):
    def __init__(self, args: CoconutMindConfig):
        super().__init__()
        # 取出词表大小和中间层大小
        self.vocab_size, self.num_hidden_layers = (
            args.vocab_size,
            args.num_hidden_layers,
        )

        # 1. input embedding
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size) # 词嵌入层, 将词ID映射到隐藏向量
        
        self.dropout = nn.Dropout(args.dropout) # dropout 层

        # transformer layers
        self.layers = nn.ModuleList( # 多层 Transformer Block, 数量和 num_hidden_layers 一致
            [CoconutMindBlock(i, args) for i in range(args.num_hidden_layers)]
        )

        # rms norm
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        # RoPE 预计算
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=args.hidden_size // args.num_attention_heads,
            end=args.max_position_embeddings,
            rope_base=args.rope_theta,
            rope_scaling=args.rope_scaling,
        )
        # 注册为 buffer
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    # 定义整个网络
    def forward(
        self,
        input_ids: torch.Tensor,  # 输入的 token IDs, shape (batch_size, seq_len)，用于映射获取 token embeddings
        attention_mask: Optional[torch.Tensor] = None, # attention mask, shape (batch_size, seq_len)，用于指定哪些 token 可以被 attention 到
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None, # 缓存的 key/value 张量, 用于加速生成
        use_cache: bool = False,
        **kwargs, # 其他参数, 为了兼容性
    ) -> torch.Tensor:
        bsz, seq_len = input_ids.size()  # 获取输入的 batch size 和序列长度

        # 兼容性处理
        if hasattr(past_key_values, "layers"):
            past_key_values = past_key_values.layers
        
        # 如果 layers 为空，初始化为 None 列表
        past_key_values = past_key_values or [None] * self.num_hidden_layers

        # 计算起始位置
        # 根据 past_key_values 的长度计算，如果有缓存，则 start_pos 是已经缓存的序列长度，否则是 0
        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )

        # 1. 获取 token embeddings
        hidden_states = self.embed_tokens(input_ids)  # (bsz, seq_len,

        # 2. dropout
        hidden_states = self.dropout(hidden_states)

        # 3. 位置编码
        position_embeddings = (self.freqs_cos[start_pos : start_pos + seq_len], self.freqs_sin[start_pos : start_pos + seq_len])

        # 4. transformer layers
        presents = [] # 用于存储每一层的 present key/value
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present_key_value = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present_key_value)

        # 5. rms norm
        hidden_states = self.norm(hidden_states)


        # 返回隐藏层输出，presents 是一个包含每一层 key/value 缓存的列表，用于生成阶段避免重复计算
        # 但是为什么没有先判断是否 use_cache 呢？ 因为即使不使用缓存，返回 presents 也不会有问题，只是不会被使用而已，这里并不占用额外内存，如果没有使用，会被回收吗 -- 会的
        return hidden_states, presents
    

# 定义最终的模型类
class CoconutMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = CoconutMindConfig
    def __init__(self, config: config_class):
        self.config = config # 需要在父类初始化之前设置 config
        super().__init__()
        self.model = CoconutMindModel(config) # 基础模型
        # lm head： 线性层，将隐藏状态映射到词表大小
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 输出层和输入层共享权重
        self.model.embed_tokens.weight = self.lm_head.weight

        # 定义输出
        # CausalLMOutputWithPast 是 transformers 提供的一个标准输出类，包含了语言模型常用的输出项
        self.OUT = CausalLMOutputWithPast()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = None, # 用于生成时限制 logits 的大小, logits 是指模型输出的未归一化概率分布
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        ) # 获取隐藏状态和缓存的 key/value

        # 如果 logits_to_keep 是整数，则保留隐藏层的最后 logits_to_keep 个维度
        # 否则，使用提供的索引张量进行切片
        # 在参数传递的时候，用的是 Union[int, torch.Tensor]，所以 logits_to_keep 要不么是整数，要么是张量
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )

        # 切片之后（截断之后），传入 lm head
        # 这里 ... 的语法表示保留前面的所有维度，只在最后一个维度上进行切片
        logits = self.lm_head(hidden_states[..., slice_indices]) # 计算 logits
        
        # 定义 out 需要输出的内容
        self.OUT.__setitem__('last_hidden_state', hidden_states)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('past_key_values', past_key_values)

        return self.OUT
    
    # 这里输出的 OUT 会再经过 tokenizer 的 decoder 变成文本
    # softmax 一般在 loss function 里计算，可以实现数值稳定性更好的交叉熵计算
