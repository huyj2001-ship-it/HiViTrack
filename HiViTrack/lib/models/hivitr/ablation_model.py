import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath, trunc_normal_

# 假设您的代码中已经定义了 Mlp, Attention, PatchEmbed, PatchMerge 等基础模块
# 如果没有，您需要从原始代码中将它们复制过来。
# 这里我们仅为示例提供一个简化的 Mlp 和 Attention 占位符
# --- 您需要用您自己的真实模块替换下面的占位符 ---
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))

class Attention(nn.Module):
    # 这是一个简化的占位符，您需要使用您项目中真实的Attention类
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        # 这是一个简化的forward，真实的可能更复杂
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # ... attention calculation ...
        # 这里省略了复杂的注意力计算，仅为结构示意
        return self.proj(x)

# 假设 PatchEmbed 和 PatchMerge 也已定义
# from your_project.models import PatchEmbed, PatchMerge, BlockWithRPE

# ---------------------------------------------------------------------------------
# 1. 为浅层阶段新建一个标准的自注意力Block (这是关键修改)
# ---------------------------------------------------------------------------------
class SelfAttentionBlock(nn.Module):
    """
    标准的Transformer Block，包含自注意力(Self-Attention)和MLP。
    专为HiViTr的浅层（孪生网络）阶段设计，其forward函数只接收一个参数x。
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # 注意：这里的Attention应该是标准的自注意力模块
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        """
        这个简单的forward签名与浅层阶段的调用方式 blk(x) 兼容。
        """
        # 注意：这里的attn调用是标准的，没有额外的 t_h, t_w 等参数
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# ---------------------------------------------------------------------------------
# 2. 修改HiViT类，以创建 "Uniform Transformer" 消融模型
# ---------------------------------------------------------------------------------
class HiViT_Ablation_SSDR(HiViT): # 继承自您原始的HiViT类
    """
    用于验证SSDR的消融模型。
    它重写了__init__方法，在浅层阶段使用SelfAttentionBlock替换原来的MLP Block。
    """
    def __init__(self, *args, **kwargs):
        # 首先调用父类的__init__来初始化所有基本组件
        super().__init__(*args, **kwargs)

        # ---- 以下是重写的核心逻辑 ----
        
        # 清空并重建 self.blocks
        self.blocks = nn.ModuleList()
        
        # 重新获取参数
        embed_dim_start = self.num_features // 2 ** (self.num_layers - 1)
        embed_dim = embed_dim_start
        depths = kwargs.get('depths', [2, 2, 20])
        num_heads = kwargs.get('num_heads', 8)
        mlp_ratio = kwargs.get('mlp_ratio', 4.)
        stem_mlp_ratio = kwargs.get('stem_mlp_ratio', 3.)
        drop_path_rate = kwargs.get('drop_path_rate', 0.0)
        norm_layer = kwargs.get('norm_layer', nn.LayerNorm)
        
        dpr = iter(x.item() for x in torch.linspace(0, drop_path_rate, sum(depths) + sum(depths[:-1])))

        # 重建 blocks 列表，应用消融逻辑
        for i, stage_depth in enumerate(depths):
            is_main_stage = (i == len(depths) - 1) # 判断是否是最后一个（深层）阶段
            
            # -- 关键修改 --
            if is_main_stage:
                # 深层阶段：保持不变，使用原来的、复杂的BlockWithRPE进行跨注意力融合
                ratio = mlp_ratio
                for _ in range(stage_depth):
                    self.blocks.append(
                        BlockWithRPE(
                            input_size=None, # 根据您的BlockWithRPE定义传入
                            dim=embed_dim,
                            num_heads=num_heads,
                            mlp_ratio=ratio,
                            # ... 传入其他所有必要的参数 ...
                            drop_path=next(dpr)
                        )
                    )
            else:
                # 浅层阶段：使用我们新定义的 SelfAttentionBlock
                ratio = stem_mlp_ratio
                # 在您的原始代码中，浅层stage的depth被乘以了2
                # stage_depth = stage_depth * 2 # 如果需要，保留这行
                for _ in range(stage_depth):
                    self.blocks.append(
                        SelfAttentionBlock(
                            dim=embed_dim,
                            # 浅层也需要指定num_heads
                            # 可以使用一个较小的值，或直接用主num_heads
                            num_heads=num_heads, 
                            mlp_ratio=ratio,
                            # ... 传入其他所有必要的参数 ...
                            drop_path=next(dpr)
                        )
                    )
            
            if not is_main_stage:
                self.blocks.append(
                    PatchMerge(embed_dim, norm_layer)
                )
                embed_dim *= 2
                
        # 确保父类的其他属性被正确初始化
        self.apply(self._init_weights)

# ---------------------------------------------------------------------------------
# 3. 为这个消融模型创建一个新的构建函数
# ---------------------------------------------------------------------------------
def build_hivitr_ablation_ssdr(cfg, training=True):
    """
    构建用于验证SSDR的 "Uniform Transformer" 消融模型。
    """
    # 假设 get_hivit_base 这样的函数返回的是一个 HiViT 实例
    # 我们需要创建一个类似的函数来返回 HiViT_Ablation_SSDR 实例
    
    backbone = HiViT_Ablation_SSDR(
        img_size_s=cfg.DATA.SEARCH.SIZE,
        img_size_t=cfg.DATA.TEMPLATE.SIZE,
        embed_dim=512, 
        depths=[2, 2, 20], 
        num_heads = 8, 
        stem_mlp_ratio=3., 
        mlp_ratio=4., 
        rpe=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop_path_rate=cfg.TRAIN.DROP_PATH_RATE
    )
    
    # --- 加载预训练权重的逻辑 (可选，但建议) ---
    # ...
    
    # 假设 Hivitr, build_box_head 已经定义
    box_head = build_box_head(cfg)
    model = Hivitr(
        backbone,
        box_head,
        head_type=cfg.MODEL.HEAD_TYPE
    )
    return model