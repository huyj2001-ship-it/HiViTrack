import math
import torch
import torch.nn as nn
from functools import partial
import torch.utils.checkpoint as checkpoint
from timm.models.vision_transformer import DropPath, Mlp, trunc_normal_
from timm.models.layers import to_2tuple
from lib.models.utils.pos_utils import get_2d_sincos_pos_embed
from .head import build_box_head
from lib.utils.misc import is_main_process
from lib.utils.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from einops import rearrange
from einops.layers.torch import Rearrange
from lib.models.hivitr.score_decoder import ScoreDecoder

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, inner_patches=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.inner_patches = inner_patches
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        conv_size = [size // inner_patches for size in patch_size]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=conv_size, stride=conv_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        patches_resolution = (H // self.patch_size[0], W // self.patch_size[1])
        num_patches = patches_resolution[0] * patches_resolution[1]
        x = self.proj(x).view(
            B, -1, 
            patches_resolution[0], self.inner_patches, 
            patches_resolution[1], self.inner_patches, 
        ).permute(0, 2, 4, 3, 5, 1).reshape(B, num_patches, self.inner_patches, self.inner_patches, -1)
        if self.norm is not None:
            x = self.norm(x)
        return x
            

class PatchMerge(nn.Module):
    def __init__(self, dim, norm_layer):
        super().__init__()
        self.norm = norm_layer(dim * 4)
        self.reduction = nn.Linear(dim * 4, dim * 2, bias=False)
    
    def forward(self, x):
        x0 = x[..., 0::2, 0::2, :] 
        x1 = x[..., 1::2, 0::2, :] 
        x2 = x[..., 0::2, 1::2, :] 
        x3 = x[..., 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., rpe=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.qkv_mem = None

    def forward(self, x, t_h, t_w, s_h, s_w):
        """
        x is a concatenated vector of template and search region features.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q_mt, q_s = torch.split(q, [t_h * t_w * 2, s_h * s_w], dim=2)
        k_mt, k_s = torch.split(k, [t_h * t_w * 2, s_h * s_w], dim=2)
        v_mt, v_s = torch.split(v, [t_h * t_w * 2, s_h * s_w], dim=2)

        # asymmetric mixed attention
        attn = (q_mt @ k_mt.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_mt = (attn @ v_mt).transpose(1, 2).reshape(B, t_h*t_w*2, C)

        attn = (q_s @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_s = (attn @ v).transpose(1, 2).reshape(B, s_h*s_w, C)

        x = torch.cat([x_mt, x_s], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def forward_test(self, x, s_h, s_w):
        """
        x is a concatenated vector of template and search region features.
        """
        B, N, C = x.shape
        qkv_s = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_s, _, _ = qkv_s.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        qkv = torch.cat([self.qkv_mem, qkv_s], dim=3)
        _, k, v = qkv.unbind(0)

        attn = (q_s @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, s_h*s_w, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def set_online(self, x, t_h, t_w):
        """
        x is a concatenated vector of template and search region features.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        self.qkv_mem = qkv
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple) [B, num_heads, N, C//num_heads]

        # asymmetric mixed attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class BlockWithRPE(nn.Module):
    def __init__(self, input_size, dim, num_heads=0., mlp_ratio=4., qkv_bias=True, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., rpe=True, 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        with_attn = num_heads > 0.

        self.norm1 = norm_layer(dim) if with_attn else None
        self.attn = (Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=drop
        ) if with_attn else None)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, t_h=None, t_w=None, s_h=None, s_w=None):
        if self.attn is not None:
            x = x + self.drop_path(self.attn(self.norm1(x), t_h, t_w, s_h, s_w))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
    def forward_test(self, x, s_h=None, s_w=None):
        if self.attn is not None:
            x = x + self.drop_path(self.attn.forward_test(self.norm1(x), s_h, s_w))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    def set_online(self, x, t_h=None, t_w=None):
        if self.attn is not None:
            x = x + self.drop_path(self.attn.set_online(self.norm1(x), t_h, t_w))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class HiViT(nn.Module):
    def __init__(self, 
                 img_size_s=256,img_size_t=128, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=512, depths=[2, 2, 20], num_heads=8, stem_mlp_ratio=3., mlp_ratio=4., 
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,
                 norm_layer=nn.LayerNorm, ape=True, rpe=True, patch_norm=True, use_checkpoint=False, 
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.ape = ape
        self.rpe = rpe
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.num_main_blocks = depths[-1]

        embed_dim = embed_dim // 2 ** (self.num_layers - 1)
        #
        self.grid_size_s = img_size_s // patch_size
        self.grid_size_t = img_size_t // patch_size
        self.num_patches_s = self.grid_size_s**2
        self.num_patches_t = self.grid_size_t**2
        self.pos_embed_s = nn.Parameter(
            torch.zeros(1, self.num_patches_s, self.num_features),
            requires_grad=False,
        )
        self.pos_embed_t = nn.Parameter(
            torch.zeros(1, self.num_patches_t, self.num_features),
            requires_grad=False,
        )
        self.init_pos_embed()
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        Hp, Wp = self.patch_embed.patches_resolution
        assert Hp == Wp

        # absolute position embedding
        if ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, self.num_features)
            )
            trunc_normal_(self.absolute_pos_embed, std=.02)
        if rpe:
            coords_h = torch.arange(Hp)
            coords_w = torch.arange(Wp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w])) 
            coords_flatten = torch.flatten(coords, 1) 
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] 
            relative_coords = relative_coords.permute(1, 2, 0).contiguous() 
            relative_coords[:, :, 0] += Hp - 1 
            relative_coords[:, :, 1] += Wp - 1
            relative_coords[:, :, 0] *= 2 * Wp - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = iter(x.item() for x in torch.linspace(0, drop_path_rate, sum(depths) + sum(depths[:-1])))

        # build blocks
        self.blocks = nn.ModuleList()
        for stage_depth in depths:
            is_main_stage = (embed_dim == self.num_features)
            nhead = num_heads if is_main_stage else 0
            ratio = mlp_ratio if is_main_stage else stem_mlp_ratio
            # every block not in main stage include two mlp blocks
            stage_depth = stage_depth if is_main_stage else stage_depth * 2
            for _ in range(stage_depth):
                self.blocks.append(
                    BlockWithRPE(
                        Hp, embed_dim, nhead, ratio, qkv_bias, qk_scale, 
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=next(dpr), 
                        rpe=rpe, norm_layer=norm_layer,
                    )
                )
            if not is_main_stage:
                self.blocks.append(
                    PatchMerge(embed_dim, norm_layer)
                )
                embed_dim *= 2

        # self.fc_norm = norm_layer(self.num_features)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}
    def init_pos_embed(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed_t = get_2d_sincos_pos_embed(
            self.pos_embed_t.shape[-1], int(self.num_patches_t**0.5), cls_token=False
        )  # [1, 8*8=64, dim], 128/16=8
        self.pos_embed_t.data.copy_(torch.from_numpy(pos_embed_t).float().unsqueeze(0))

        pos_embed_s = get_2d_sincos_pos_embed(
            self.pos_embed_s.shape[-1], int(self.num_patches_s**0.5), cls_token=False
        )  # [1, 16*16=256, dim], 256/16=16
        self.pos_embed_s.data.copy_(torch.from_numpy(pos_embed_s).float().unsqueeze(0))

    def forward(self, x_t, x_ot, x_s):
        """
        :param x_t: (batch, c, 128, 128)
        :param x_s: (batch, c, 288, 288)
        :return:
        """
        x_t = self.patch_embed(x_t)
        x_ot = self.patch_embed(x_ot)
        x_s = self.patch_embed(x_s)
        
        for blk in self.blocks[: -self.num_main_blocks]:
            x_t = blk(x_t)
            x_ot = blk(x_ot)
            x_s = blk(x_s)
            
        x_t = x_t[..., 0, 0, :]
        x_ot = x_ot[..., 0, 0, :]
        x_s = x_s[..., 0, 0, :]
        
        B, C = x_t.size(0), x_t.size(-1)
        H_s = W_s = int(math.sqrt(x_s.size(1)))
        H_t = W_t = int(math.sqrt(x_t.size(1)))

        x_s = x_s + self.pos_embed_s
        x_t = x_t + self.pos_embed_t
        x_ot = x_ot + self.pos_embed_t
        x = torch.cat([x_t, x_ot, x_s], dim=1)
        x = self.pos_drop(x)

        for blk in self.blocks[-self.num_main_blocks :]:
            x = blk(x, H_t, W_t, H_s, W_s)

        x_t, x_ot, x_s = torch.split(x, [H_t * W_t, H_t * W_t, H_s * W_s], dim=1)

        x_t_2d = x_t.transpose(1, 2).reshape(B, C, H_t, W_t)
        x_ot_2d = x_ot.transpose(1, 2).reshape(B, C, H_t, W_t)
        x_s_2d = x_s.transpose(1, 2).reshape(B, C, H_s, W_s)

        return x_t_2d, x_ot_2d, x_s_2d
    
    def forward_test(self, x):
        x = self.patch_embed(x)

        for blk in self.blocks[: -self.num_main_blocks]:
            x = blk(x)

        x = x[..., 0, 0, :]

        H_s = W_s = int(math.sqrt(x.size(1)))

        x = x + self.pos_embed_s
        x = self.pos_drop(x)

        for blk in self.blocks[-self.num_main_blocks :]:
            x = blk.forward_test(x, H_s, W_s)

        x = rearrange(x, "b (h w) c -> b c h w", h=H_s, w=H_s)

        return self.template, x

    def set_online(self, x_t, x_ot):
        x_t = self.patch_embed(x_t)
        x_ot = self.patch_embed(x_ot)

        for blk in self.blocks[: -self.num_main_blocks]:
            x_t = blk(x_t)
            x_ot = blk(x_ot)

        x_t = x_t[..., 0, 0, :]
        x_ot = x_ot[..., 0, 0, :]

        H_t = W_t = int(math.sqrt(x_t.size(1)))

        x_t = x_t + self.pos_embed_t
        x_ot = x_ot + self.pos_embed_t
        x_ot = x_ot.reshape(1, -1, x_ot.size(-1))  # [1, num_ot * H_t * W_t, C]
        x = torch.cat([x_t, x_ot], dim=1)

        x = self.pos_drop(x)

        for blk in self.blocks[-self.num_main_blocks :]:
            x = blk.set_online(x, H_t, W_t)

        x_t = x[:, : H_t * W_t]
        x_t = rearrange(x_t, "b (h w) c -> b c h w", h=H_t, w=W_t)

        self.template = x_t

# Model PARAMs 66.42M, FLOPs 15.85G with 224 input
def get_hivit_base(cfg, training=True):
    pretrained = cfg.MODEL.PRETRAIN_FILE
    model = HiViT(
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
    if training==False and pretrained:
        ckpt = torch.load(pretrained, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
        if is_main_process():
            print("Load pretrained backbone from: " + pretrained)
            print("missing keys:", missing_keys)
            print("unexpected keys:", unexpected_keys)
            print("Loading pretrained HiVit done.")
    
    if training and pretrained:
        ckpt = torch.load(pretrained, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
        if is_main_process():
            print("Load pretrained backbone from: " + pretrained)
            print("missing keys:", missing_keys)
            print("unexpected keys:", unexpected_keys)
            print("Loading pretrained HiVit done.")
    return model

class Hivitr_Online(nn.Module):
    """ This is the base class for Transformer Tracking, whcih jointly perform feature extraction and interaction. """
    def __init__(self, backbone, box_head, score_branch=None, head_type="CORNER"):
        """ Initializes the model.
        """
        super().__init__()
        self.backbone = backbone
        self.box_head = box_head
        self.head_type = head_type
        self.score_branch = score_branch

    def forward(self, template, online_template, search, run_score_head=True, gt_bboxes=None):
        # search: (b, c, h, w)
        if template.dim() == 5:
            template = template.squeeze(0)
        if online_template.dim() == 5:
            online_template = online_template.squeeze(0)
        if search.dim() == 5:
            search = search.squeeze(0)
        template,template_online,search = self.backbone(template, online_template, search)
        # Forward the corner head
        out, outputs_coord_new = self.forward_head(search, template, run_score_head, gt_bboxes)
        return out, outputs_coord_new
    
    def forward_test(self, search, run_score_head=True,  gt_bboxes=None):
        # search: (b, c, h, w) h=20
        if search.dim() == 5:
            search = search.squeeze(0)
        template,search = self.backbone.forward_test(search)
        # search (b, 384, 20, 20)
        # Forward the corner head
        out, outputs_coord_new = self.forward_head(search, template, run_score_head, gt_bboxes)
        return out, outputs_coord_new

    def set_online(self, template, online_template):
        if template.dim() == 5:
            template = template.squeeze(0)
        if online_template.dim() == 5:
            online_template = online_template.squeeze(0)
        self.backbone.set_online(template, online_template)

    def forward_head(self, search,template, run_score_head=True, gt_bboxes=None):
        """
        :param search: (b, c, h, w)
        :return:
        """
        out_dict = {}
        out_dict_box, outputs_coord = self.forward_box_head(search)
        out_dict.update(out_dict_box)
        if run_score_head:
            # forward the classification head
            if gt_bboxes is None:
                gt_bboxes = box_cxcywh_to_xyxy(outputs_coord.clone().view(-1, 4))
            # (b,c,h,w) --> (b,h,w)
            out_dict.update({'pred_scores': self.score_branch(search, template, gt_bboxes).view(-1)})

        return out_dict, outputs_coord
    
    def forward_box_head(self, search):
        """
        :param search: (b, c, h, w)
        :return:
        """
        if "CORNER" in self.head_type:
            # run the corner head
            b = search.size(0)
            outputs_coord = box_xyxy_to_cxcywh(self.box_head(search))
            outputs_coord_new = outputs_coord.view(b, 1, 4)
            out = {'pred_boxes': outputs_coord_new}
            return out, outputs_coord_new
        else:
            raise KeyError

def build_hivitr_online(cfg,settings=None,training=True):
    backbone = get_hivit_base(cfg,training=True)  # backbone without positional encoding and attention mask
    score_branch = ScoreDecoder(pool_size=4, hidden_dim=cfg.MODEL.HIDDEN_DIM, num_heads=cfg.MODEL.HIDDEN_DIM//64)  # the proposed score prediction module (SPM)
    box_head = build_box_head(cfg)  # a simple corner head
    model = Hivitr_Online(
        backbone,
        box_head,
        score_branch,
        head_type=cfg.MODEL.HEAD_TYPE
    )
    if cfg.MODEL.PRETRAINED_STAGE1 and training:
        ckpt_path = settings.stage1_model
        print(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(ckpt['net'], strict=False)
        if is_main_process():
            print("missing keys:", missing_keys)
            print("unexpected keys:", unexpected_keys)
            print("Loading pretrained hivit weights done.")        
    return model