
import torch
import torch.nn as nn
from functools import partial

import torch.nn.functional as F
import numpy as np

from models.vit_utils import DropPath, to_2tuple, trunc_normal_

from torch import einsum
from einops import rearrange, reduce, repeat
NUM_FRAMES = 8


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x


class mask_Attention(nn.Module):
    """
    A customized attention module that applies three separate masks (head/body/interact)
    to decompose attention into region-specific interactions.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Initialize attention masks for each interaction type
        matrix = torch.zeros((B,self.num_heads,N, N)).to(x.device)
        matrix [ :,:, 0, 0] = 1

        head_masked = matrix.clone()
        body_masked = matrix.clone()
        interact_masked = matrix.clone()

        # Define attention regions
        head_masked[ :,:,:NUM_FRAMES + 1, :NUM_FRAMES + 1] = 1
        body_masked[  :,:,NUM_FRAMES + 1:, NUM_FRAMES + 1:] = 1
        body_masked[  :,:,0, NUM_FRAMES + 1:] = 1
        body_masked[  :,:,NUM_FRAMES + 1:, 0] = 1
        interact_masked[ :,:, NUM_FRAMES + 1:, :NUM_FRAMES + 1] = 1
        interact_masked[ :,:,:NUM_FRAMES + 1, NUM_FRAMES + 1:] = 1

        # Apply masks
        head_scores = attn.masked_fill( head_masked == 0, -1e9)
        body_scores = attn.masked_fill(body_masked == 0, -1e9)
        interact_scores = attn.masked_fill(interact_masked == 0, -1e9)

        head_scores = self.attn_drop(head_scores.softmax(dim=-1))
        body_scores = self.attn_drop(body_scores.softmax(dim=-1))
        interact_scores = self.attn_drop(interact_scores.softmax(dim=-1))

        x_head = (head_scores @ v).transpose(1, 2).reshape(B, N, C)
        x_body = (body_scores @ v).transpose(1, 2).reshape(B, N, C)
        x_interact = (interact_scores @ v).transpose(1, 2).reshape(B, N, C)

        x_head = self.proj_drop(self.proj(x_head))
        x_body = self.proj_drop(self.proj(x_body))
        x_interact = self.proj_drop(self.proj(x_interact))

        return x_head,x_body,x_interact



class sep_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv
        attn = (q @ k.transpose(-2, -1)) * self.scale

        matrix = torch.zeros((B,self.num_heads,N, N)).to(x.device)
        matrix [ :,:, 0, 0] = 1
        interact_masked = matrix.clone()
        interact_masked[ :,:, NUM_FRAMES + 1:, :NUM_FRAMES + 1] = 1
        interact_masked[ :,:,:NUM_FRAMES + 1, NUM_FRAMES + 1:] = 1

        interact_scores = attn.masked_fill(interact_masked == 0, -1e9)
        interact_scores = self.attn_drop(interact_scores.softmax(dim=-1))
        x_interact = (interact_scores @ v).transpose(1, 2).reshape(B, N, C)
        x_interact = self.proj_drop(self.proj(x_interact))
        return x_interact


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.mask_attn = sep_Attention(
              dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x ,parts):
        if parts == 'interact':
            x=self.mask_attn(self.norm2(x))
        else:
            x = self.attn(self.norm2(x))
        xt = self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(xt)))
        return x


class sep_Block(nn.Module):
    """
    Relation-Block
    Transformer block that separates feature processing into three parts:
    head, body, and interaction — using mask_Attention module.
    Each branch passes through attention + FFN independently.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer(dim)
        # Use the custom masked attention that separates head/body/interac
        self.mask_attn = mask_Attention(
              dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):

        head_res,body_res,interact_res=self.mask_attn(self.norm2(x))
        head_res = self.drop_path(head_res)
        body_res = self.drop_path(body_res)
        interact_res = self.drop_path(interact_res)

        head_res = head_res + self.drop_path(self.mlp(self.norm2(head_res)))
        body_res = body_res + self.drop_path(self.mlp(self.norm2(body_res)))
        interact_res = interact_res + self.drop_path(self.mlp(self.norm2(interact_res)))

        return head_res,body_res,interact_res


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding Origin
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, T, P, H, W = x.shape
        x = rearrange(x, 'b c t p h w -> (b t) (c h w) p ')

        W = x.size(-1)
        x = x.flatten(2).transpose(1, 2)
        return x, T, W

class conv2d_block(nn.Module):
    def __init__(self, in_channels, out_channels, pad='same', k=3, s=1):
        super(conv2d_block, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=pad, stride=s, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x


class cnn_face(nn.Module):
    def __init__(self, pool = False):
        super(cnn_face, self).__init__()
        self.pool = pool
        self.conv1 = conv2d_block(3, 64, k=7, pad=(3, 3), s=2)
        self.layer1 = nn.Sequential(
            conv2d_block(64, 64),
            conv2d_block(64, 64),
        )

        self.conv2 = conv2d_block(64, 128, k=3, pad=(1, 1), s=2)
        self.layer2 = nn.Sequential(
            conv2d_block(128, 128),
            conv2d_block(128, 128),
        )

        self.conv3 = conv2d_block(128, 256, k=3, pad=(1, 1), s=2)
        self.layer3 = nn.Sequential(
            conv2d_block(256, 256),
            conv2d_block(256, 256),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x) + x
        x = self.conv2(x)
        x = self.layer2(x) + x
        x = self.conv3(x)
        x = self.layer3(x) + x
        if self.pool:
            x = self.avg_pool(x)
        return x

class PatchEmbed_x(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = cnn_face()

    def forward(self, x):
        B, C, T, P, H, W = x.shape
        x = rearrange(x, 'b c t p h w -> (b t p) c h w')
        x = self.proj(x)
        W = x.size(-1)
        x = x.flatten(2)
        x = rearrange(x, '(b t p) m n -> (b t) p m n', b = B, p = P, t = T)
        x = x.transpose(-2, -1)
        return x, T, W

class Fusion(nn.Module):
    """
    Fusion module for aggregating features across patches tokens using attention.
    
    Key steps:
    - Projects input into QKV triplet
    - Computes cross-attention between query and key
    - Applies 2D convolution + batch norm over attention matrix
    - Uses softmaxed attention to aggregate value features into a compressed representation
    
    """
    def __init__(self, dim, N, num_heads=8, num_patches=16, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.conv = nn.Conv2d(num_heads, num_heads, kernel_size=(N, 1), stride=1)
        self.bn = nn.BatchNorm2d(num_heads)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    def forward(self, x, B, T, W):

        _, _, C = x.shape

        qkv = self.qkv(x).reshape(B*T, W*W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute scaled dot-product attention between K and Q (note: transpose order)
        attn = (k @ q.transpose(-2, -1))  # [B*T, num_heads, N, N]

        # Apply convolution along token dimension (N), output shape: [B*T, num_heads, 1, N]
        attn = self.conv(attn)
        attn = self.bn(attn)

        # Normalize across last dimension to get attention weights
        attn_mul = attn.softmax(dim=-1)  # [B*T, num_heads, 1, N]

        # Weighted sum of values using attention
        x = (attn_mul @ v).transpose(1, 2).reshape(B*T, 1, C)  # output fused feature [B*T, 1, C]
        return x, attn_mul


class VisionTransformer(nn.Module):
    """ Vision Transformer Design for Deception Detection
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_patches = 5,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, num_frames=8, dropout=0.):
        super().__init__()
        torch.manual_seed(1111)
        torch.cuda.manual_seed_all(1111)

        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed_x(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.fusion_model = Fusion(dim=embed_dim, N=(img_size//8)**2, num_heads=num_heads)

        self.num_patches = num_patches
        global NUM_FRAMES
        NUM_FRAMES = num_frames
        ## Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.time_drop = nn.Dropout(p=drop_rate)

        ## Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule

        self.interact_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(self.depth)])


        self.head_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(self.depth)])

        self.body_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(self.depth)])


        self.RAB_blk=sep_Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,)

        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
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
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        # B*T, P, H*W, C (16*8, 5, 16, 256)
        x, T, W = self.patch_embed(x) 
        # cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1) # B*T, N+1, C
        x_g = None
        for i in range(self.num_patches):
            x_temp = x[:,i,:,:]
            #x_temp = torch.unsqueeze(x_temp, dim=-3)
            x_blk,_ = self.fusion_model(x_temp, B, T, W)
            if x_g is None:
                x_g = x_blk
            else:
                x_g = torch.cat([x_g,x_blk], dim=1)
            
        x = x_g + self.pos_embed
        x = self.pos_drop(x)

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # B*T, N+1, C
        ## Time Embeddings

        cls_tokens = x[:B, 0, :].unsqueeze(1)
        x = x[:,1:]
        x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)
        x = x + self.time_embed
        x = self.time_drop(x)
        x = rearrange(x, '(b n) t m -> b (n t) m',b=B,t=T)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x_head, x_body, x_interact = self.RAB_blk(x)
        for blk in self.head_blocks:
            x_head = blk(x_head,'head')
        for blk in self.interact_blocks:
            x_interact = blk(x_interact,'interact')
        for blk in self.body_blocks:
            x_body = blk(x_body,'body')


        x_head = self.norm(x_head)
        x_body = self.norm(x_body)
        x_interact = self.norm(x_interact)
        return x_head[:, 0],x_body[:, 0],x_interact[:, 0]

    def forward(self, x):
        x_head,x_body,x_interact = self.forward_features(x)
        x_head = self.head(x_head)
        x_body = self.head(x_body)
        x_interact = self.head(x_interact)
        return x_head,x_body,x_interact


class DLF_BRAM_NET(nn.Module):

    def __init__(self, img_size=224, patch_size=16, num_classes=400, num_patches = 5, num_frames=8, depth = 4, **kwargs):
        super(DLF_BRAM_NET, self).__init__()
        torch.manual_seed(1111)
        torch.cuda.manual_seed_all(1111)

        self.model = VisionTransformer(img_size=img_size, num_classes=num_classes, patch_size=patch_size, embed_dim=256, 
                                       depth=depth, num_patches = num_patches, num_heads=8, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                                       drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=num_frames, **kwargs)

    def forward(self, x):
        x1,x2,x3 = self.model(x)

        return x1,x2,x3
