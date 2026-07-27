import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
from timm.models.layers import DropPath, trunc_normal_
# from model.PCT_Transformer_encoder import PCTransformer_encoder
import math
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange

# 只保留swin unet相关代码，只做预报


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
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


def window_partition(x: torch.Tensor, window_size: tuple):
    """
    Args:
        x: (B, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1],
               window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4,
                        5).contiguous().view(-1, window_size[0],
                                             window_size[1], C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1],
                     window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self,
                 dim,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 earth_specific_pos=False):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        if earth_specific_pos:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    window_size[0] * window_size[0] * (2 * window_size[1] - 1),
                    num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        else:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                    num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        if earth_specific_pos:
            coords_hi = torch.arange(self.window_size[0])
            coords_hj = -torch.arange(
                self.window_size[0]) * self.window_size[0]
            coords_w = torch.arange(self.window_size[1])
            coords_1 = torch.stack(torch.meshgrid([coords_hi, coords_w]))
            coords_2 = torch.stack(torch.meshgrid([coords_hj, coords_w]))
            coords_flatten_1 = torch.flatten(coords_1, 1)
            coords_flatten_2 = torch.flatten(coords_2, 1)
            coords = coords_flatten_1[:, :, None] - coords_flatten_2[:,
                                                                     None, :]
            coords = coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            coords[:, :, 1] += self.window_size[1] - 1
            coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = coords.sum(-1)  # Wh*Ww, Wh*Ww

        else:
            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h,
                                                 coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :,
                                             None] - coords_flatten[:,
                                                                    None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(
                1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.window_size[
                0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

        self.register_buffer("relative_position_index",
                             relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)]

        relative_position_bias = relative_position_bias.view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=(7, 7),
                 shift_size=(0, 0),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 fused_window_process=False,
                 earth_specific_pos=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if input_resolution[0] <= self.window_size[0] or input_resolution[
                1] <= self.window_size[1]:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = [0, 0]
            self.window_size = self.input_resolution
        assert 0 <= self.shift_size[0] < self.window_size[
            0] and 0 <= self.shift_size[1] < self.window_size[
                1], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim,
                                    window_size=self.window_size,
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    attn_drop=attn_drop,
                                    proj_drop=drop,
                                    earth_specific_pos=earth_specific_pos)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

        if self.shift_size[0] > 0 and self.shift_size[1] > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], -self.shift_size[0]),
                        slice(-self.shift_size[0], None))
            w_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size[1]),
                        slice(-self.shift_size[1], None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(
                img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(
                -1, self.window_size[0] * self.window_size[1])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                                                  attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size[0] > 0 and self.shift_size[1] > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x,
                                       shifts=(-self.shift_size[0],
                                               -self.shift_size[1]),
                                       dims=(1, 2))
                # partition windows
                x_windows = window_partition(
                    shifted_x,
                    self.window_size)  # nW*B, window_size, window_size, C
            else:
                x_windows = WindowProcess.apply(x, B, H, W, C,
                                                -self.shift_size,
                                                self.window_size)
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(
                shifted_x,
                self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1,
                                   self.window_size[0] * self.window_size[1],
                                   C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(
            x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0],
                                         self.window_size[1], C)

        # reverse cyclic shift
        if self.shift_size[0] > 0 and self.shift_size[1] > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H,
                                           W)  # B H' W' C
                x = torch.roll(shifted_x,
                               shifts=(self.shift_size[0], self.shift_size[1]),
                               dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C,
                                               self.shift_size,
                                               self.window_size)
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H,
                                       W)  # B H' W' C
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpand(nn.Module):

    def __init__(self,
                 input_resolution,
                 dim,
                 dim_scale=2,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(
            dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x,
                      'b h w (p1 p2 c)-> b (h p1) (w p2) c',
                      p1=2,
                      p2=2,
                      c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 earth_specific_pos=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0) if
                (i % 2 == 0) else [window_size[0] // 2, window_size[1] // 2],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i]
                if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                earth_specific_pos=earth_specific_pos) for i in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class FinalPatchExpand_xP(nn.Module):
    '''
    Up-Sampling the tokens with Patch Size to align the input image resolution. 
    https://github.com/HuCaoFighting/Swin-Unet/blob/main/networks/swin_transformer_unet_skip_expand_decoder_sys.py
    '''

    def __init__(self, input_resolution, dim, dim_scale=4, upscale=4):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.output_dim = dim
        self.upscale = upscale

        self.conv_after_body = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1))

        self.conv_before_upsample = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1),
                                                  nn.LeakyReLU(inplace=True))
        self.conv_up1 = nn.Conv2d(dim, dim, 3, 1, 1)

        if self.upscale == 4:
            self.conv_up2 = nn.Conv2d(dim, dim, 3, 1, 1)

        self.conv_last = nn.Conv2d(dim, self.output_dim, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv_hr = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C).contiguous()
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.conv_after_body(x)
        x = self.conv_before_upsample(x)
        x = self.lrelu(
            self.conv_up1(
                torch.nn.functional.interpolate(x,
                                                scale_factor=2,
                                                mode='nearest')))
        if self.upscale == 4:
            x = self.lrelu(
                self.conv_up2(
                    torch.nn.functional.interpolate(x,
                                                    scale_factor=2,
                                                    mode='nearest')))
        x = self.conv_last(self.lrelu(self.conv_hr(x)))

        return x


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. 
        patch_size (int | tuple(int)): Patch size. 
        in_chans (int): Number of input image channels. 
        embed_dim (int): Patch embedding dimension. 
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (tuple[int]): Window size. 
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. 
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
    """

    def __init__(self,
                 params,
                 img_size=[720, 1440],
                 patch_size=2,
                 in_chans=96,
                 out_chans=94,
                 embed_dim=192,
                 depths=[1, 2, 2, 2, 1],
                 num_heads=[3, 6, 6, 6, 3],
                 window_size=[6, 12],
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 earth_specific_pos=False):
        super().__init__()

        self.img_size = params['img_size']
        self.in_chans = params['net_image']['N_in_channels']
        self.out_chans = params['net_image']['N_out_channels']
        self.depths = params['net_image']['depths']
        self.num_layers = len(self.depths)
        self.num_heads = params['net_image']['num_heads']
        self.embed_dim = params['net_image']['embed_dim']
        self.window_size = params['net_image']['window_size']
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(self.embed_dim * 2**(self.num_layers - 1))
        self.mlp_ratio = params['net_image']['mlp_ratio']
        self.patch_size = params['net_image']['patch_size']
        self.earth_specific_pos = params['net_image']['earth_specific_pos']

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]  # stochastic depth decay rule

        self.layer0 = BasicLayer(
            dim=self.embed_dim,
            input_resolution=(patches_resolution[0], patches_resolution[1]),
            depth=self.depths[0],
            num_heads=self.num_heads[0],
            window_size=self.window_size,
            mlp_ratio=self.mlp_ratio,
            drop_path=dpr[sum(self.depths[:0]):sum(self.depths[:1])],
            earth_specific_pos=self.earth_specific_pos)

        self.downsample = PatchMerging(
            (patches_resolution[0], patches_resolution[1]),
            dim=self.embed_dim,
            norm_layer=norm_layer)

        self.layer1 = BasicLayer(
            dim=self.embed_dim * 2,
            input_resolution=(patches_resolution[0] // 2,
                              patches_resolution[1] // 2),
            depth=self.depths[1],
            num_heads=self.num_heads[1],
            window_size=self.window_size,
            mlp_ratio=self.mlp_ratio,
            drop_path=dpr[sum(self.depths[:1]):sum(self.depths[:2])],
            earth_specific_pos=self.earth_specific_pos)

        self.layer2 = BasicLayer(
            dim=self.embed_dim * 2,
            input_resolution=(patches_resolution[0] // 2,
                              patches_resolution[1] // 2),
            depth=self.depths[2],
            num_heads=self.num_heads[2],
            window_size=self.window_size,
            mlp_ratio=self.mlp_ratio,
            drop_path=dpr[sum(self.depths[:2]):sum(self.depths[:3])],
            earth_specific_pos=self.earth_specific_pos)

        self.layer3 = BasicLayer(
            dim=self.embed_dim * 2,
            input_resolution=(patches_resolution[0] // 2,
                              patches_resolution[1] // 2),
            depth=self.depths[3],
            num_heads=self.num_heads[3],
            window_size=self.window_size,
            mlp_ratio=self.mlp_ratio,
            drop_path=dpr[sum(self.depths[:3]):sum(self.depths[:4])],
            earth_specific_pos=self.earth_specific_pos)

        self.upsample = PatchExpand(
            (patches_resolution[0] // 2, patches_resolution[1] // 2),
            dim=self.embed_dim * 2,
            norm_layer=norm_layer)

        self.layer4 = BasicLayer(
            dim=self.embed_dim,
            input_resolution=(patches_resolution[0], patches_resolution[1]),
            depth=self.depths[4],
            num_heads=self.num_heads[4],
            window_size=self.window_size,
            mlp_ratio=self.mlp_ratio,
            drop_path=dpr[sum(self.depths[:4]):sum(self.depths[:5])],
            earth_specific_pos=self.earth_specific_pos)

        self.up = FinalPatchExpand_xP(
            input_resolution=(self.img_size[0] // self.patch_size,
                              self.img_size[1] // self.patch_size),
            dim_scale=self.patch_size,
            dim=self.embed_dim,
            upscale=self.patch_size)
        self.output = nn.Conv2d(in_channels=self.embed_dim,
                                out_channels=self.out_chans,
                                kernel_size=1,
                                bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        x = self.layer0(x)
        skip = x
        x = self.downsample(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # 保存 Encoder Output (Bottleneck)
        encoder_feature = x
        
        x = self.upsample(x)
        x = skip + x
        x = self.layer4(x)

        # feature = x # 原来的 feature 是 layer4 输出

        # 处理 encoder_feature 用于返回
        # layer3 输出分辨率是 patches_resolution 的一半 (因为经过了一次 downsample)
        H_enc, W_enc = self.patches_resolution[0] // 2, self.patches_resolution[1] // 2
        B, L, C_enc = encoder_feature.shape
        assert L == H_enc * W_enc, f"encoder feature has wrong size: {L} != {H_enc}*{W_enc}"

        encoder_feature = encoder_feature.view(B, H_enc, W_enc, C_enc).contiguous()
        encoder_feature = encoder_feature.permute(0, 3, 1, 2).contiguous() # (B, 2*embed_dim, H/4, W/4)

        x = self.up(x)
        x = self.output(x)
        output = x

        return encoder_feature, output


if __name__ == "__main__":

    # class params_naive:

    #     def __init__(self):

    #         self.img_size = (720, 960)  # 1/4 deg
    #         # self.img_size = (2040, 4320) # 1/12 deg
    #         self.patch_size = 2  # 2 / 4/ 8
    #         self.N_in_channels = 96
    #         self.N_out_channels = 94
    #         self.embed_dim = 48
    #         self.depths = [1, 2, 2, 2, 1]
    #         self.num_heads = [3, 6, 6, 6, 3]
    #         self.window_size = (
    #             7, 10
    #         )  #按照pangu whether的设定，分辨率为(720, 1440)时默认window_size = (6, 12)
    #         self.mlp_ratio = 4.
    #         self.qkv_bias = True
    #         self.qk_scale = None
    #         self.ape = False
    #         self.patch_norm = True
    #         # Dropout rate
    #         self.drop_rate = 0.0
    #         # Drop path rate
    #         self.drop_path_rate = 0.1
    #         self.fused_window_process = False
    #         self.layernorm = nn.LayerNorm
    #         self.earth_specific_pos = True

    # params = params_naive()

    # model = SwinTransformer(params,
    #                         img_size=params.img_size,
    #                         patch_size=params.patch_size,
    #                         in_chans=params.N_in_channels,
    #                         out_chans=params.N_out_channels,
    #                         embed_dim=params.embed_dim,
    #                         depths=params.depths,
    #                         num_heads=params.num_heads,
    #                         window_size=params.window_size,
    #                         mlp_ratio=params.mlp_ratio,
    #                         qkv_bias=params.qkv_bias,
    #                         qk_scale=params.qk_scale,
    #                         drop_rate=params.drop_rate,
    #                         drop_path_rate=params.drop_path_rate,
    #                         ape=params.ape,
    #                         norm_layer=params.layernorm,
    #                         patch_norm=params.patch_norm,
    #                         earth_specific_pos=params.earth_specific_pos)

    # # print(model)

    # sample = torch.randn(1, params.N_in_channels, params.img_size[0],
    #                      params.img_size[1])

    # model = model.cuda()
    # sample = sample.float().cuda()

    # result = model.forward(sample)
    # print(result.shape)

    params = {
        'img_size': (720, 960),
        'net_image': {
            'N_in_channels': 32,
            'N_out_channels': 32,
            'embed_dim': 192,
            'depths': [1, 2, 2, 2, 1],
            'num_heads': [3, 6, 6, 6, 3],
            'window_size': (6, 12),
            'mlp_ratio': 4.0,
            'patch_size': 2,
            'earth_specific_pos': False,
        }
    }

    model = SwinTransformer(params=params)

    x = torch.randn(1, 32, 720, 960)

    feature, output = model(x)

    print('feature shape:', feature.shape)
    print('output shape:', output.shape)
