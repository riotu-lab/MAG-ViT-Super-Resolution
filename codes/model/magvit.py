from model import common

import torch.nn as nn

""" MaxViT
A PyTorch implementation of the paper: `MaxViT: Multi-Axis Vision Transformer`
    - MaxViT: Multi-Axis Vision Transformer
Copyright (c) 2021 Christoph Reich
Licensed under The MIT License [see LICENSE for details]
Written by Christoph Reich
"""
from timm.layers.halo_attn import HaloAttn

from typing import Type, Callable, Tuple, Optional, Set, List, Union

import torch
from timm.layers.gather_excite import GatherExcite
import torch.nn as nn
# from timm.models.layers.eca import CecaModule
# from timm.models.layers.bottleneck_attn import BottleneckAttn
from timm.layers import SqueezeExcite
from timm.models._efficientnet_blocks import DepthwiseSeparableConv
# from timm.models.efficientnet_blocks import SqueezeExcite, DepthwiseSeparableConv
from timm.layers import drop_path, trunc_normal_, Mlp, DropPath


def _gelu_ignore_parameters(
        *args,
        **kwargs
) -> nn.Module:
    """ Bad trick to ignore the inplace=True argument in the DepthwiseSeparableConv of Timm.
    Args:
        *args: Ignored.
        **kwargs: Ignored.
    Returns:
        activation (nn.Module): GELU activation function.
    """
    activation = nn.GELU()
    return activation


class MBConv(nn.Module):
    """ MBConv block as described in: https://arxiv.org/pdf/2204.01697.pdf.
        Without downsampling:
        x ← x + Proj(SE(DWConv(Conv(Norm(x)))))
        With downsampling:
        x ← Proj(Pool2D(x)) + Proj(SE(DWConv ↓(Conv(Norm(x))))).
        Conv is a 1 X 1 convolution followed by a Batch Normalization layer and a GELU activation.
        SE is the Squeeze-Excitation layer.
        Proj is the shrink 1 X 1 convolution.
        Note: This implementation differs slightly from the original MobileNet implementation!
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        downscale (bool, optional): If true downscale by a factor of two is performed. Default: False
        act_layer (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
        norm_layer (Type[nn.Module], optional): Type of normalization layer to be utilized. Default: nn.BatchNorm2d
        drop_path (float, optional): Dropout rate to be applied during training. Default 0.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            downscale: bool = False,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            drop_path: float = 0.,
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(MBConv, self).__init__()
        # Save parameter
        self.drop_path_rate: float = drop_path
        # Check parameters for downscaling
        if not downscale:
            assert in_channels == out_channels, "If downscaling is utilized input and output channels must be equal."
        # Ignore inplace parameter if GELU is used
        if act_layer == nn.GELU:
            act_layer = _gelu_ignore_parameters
        # Make main path
        self.main_path = nn.Sequential(
            norm_layer(in_channels),
            DepthwiseSeparableConv(in_chs=in_channels, out_chs=out_channels, stride=2 if downscale else 1,
                                   act_layer=act_layer, norm_layer=norm_layer, drop_path_rate=drop_path),
            # SqueezeExcite(in_chs=out_channels, rd_ratio=0.25),
            # CbamModule(channels=out_channels, rd_ratio=0.25),
            # BottleneckAttn(dim = out_channels, feat_size = (48,48)),
            # CecaModule(channels = out_channels),
            # GatherExcite(channels = out_channels,rd_ratio=0.25),
            HaloAttn(dim = out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels
                      , kernel_size=(1, 1)), nn.Sigmoid()
        )
        # Make skip path
        self.skip_path = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))
        ) if downscale else nn.Identity()

    def forward(
            self,
            input: torch.Tensor
    ) -> torch.Tensor:
        """ Forward pass.
        Args:
            input (torch.Tensor): Input tensor of the shape [B, C_in, H, W].
        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C_out, H (// 2), W (// 2)] (downscaling is optional).
        """
        output = self.main_path(input)
        if self.drop_path_rate > 0.:
            output = drop_path(output, self.drop_path_rate, self.training)
        output = output + self.skip_path(input)
        return output


def window_partition(
        input: torch.Tensor,
        window_size: Tuple[int, int] = (7, 7)
) -> torch.Tensor:
    """ Window partition function.
    Args:
        input (torch.Tensor): Input tensor of the shape [B, C, H, W].
        window_size (Tuple[int, int], optional): Window size to be applied. Default (7, 7)
    Returns:
        windows (torch.Tensor): Unfolded input tensor of the shape [B * windows, window_size[0], window_size[1], C].
    """
    # Get size of input
    B, C, H, W = input.shape
    # Unfold input
    windows = input.view(B, C, H // window_size[0], window_size[0], W // window_size[1], window_size[1])
    # Permute and reshape to [B * windows, window_size[0], window_size[1], channels]
    windows = windows.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(
        windows: torch.Tensor,
        original_size: Tuple[int, int],
        window_size: Tuple[int, int] = (7, 7)
) -> torch.Tensor:
    """ Reverses the window partition.
    Args:
        windows (torch.Tensor): Window tensor of the shape [B * windows, window_size[0], window_size[1], C].
        original_size (Tuple[int, int]): Original shape.
        window_size (Tuple[int, int], optional): Window size which have been applied. Default (7, 7)
    Returns:
        output (torch.Tensor): Folded output tensor of the shape [B, C, original_size[0], original_size[1]].
    """
    # Get height and width
    H, W = original_size
    # Compute original batch size
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    # Fold grid tensor
    output = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    output = output.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
    return output


def grid_partition(
        input: torch.Tensor,
        grid_size: Tuple[int, int] = (7, 7)
) -> torch.Tensor:
    """ Grid partition function.
    Args:
        input (torch.Tensor): Input tensor of the shape [B, C, H, W].
        grid_size (Tuple[int, int], optional): Grid size to be applied. Default (7, 7)
    Returns:
        grid (torch.Tensor): Unfolded input tensor of the shape [B * grids, grid_size[0], grid_size[1], C].
    """
    # Get size of input
    B, C, H, W = input.shape
    # Unfold input
    grid = input.view(B, C, grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1])
    # Permute and reshape [B * (H // grid_size[0]) * (W // grid_size[1]), grid_size[0], window_size[1], C]
    grid = grid.permute(0, 3, 5, 2, 4, 1).contiguous().view(-1, grid_size[0], grid_size[1], C)
    return grid


def grid_reverse(
        grid: torch.Tensor,
        original_size: Tuple[int, int],
        grid_size: Tuple[int, int] = (7, 7)
) -> torch.Tensor:
    """ Reverses the grid partition.
    Args:
        Grid (torch.Tensor): Grid tensor of the shape [B * grids, grid_size[0], grid_size[1], C].
        original_size (Tuple[int, int]): Original shape.
        grid_size (Tuple[int, int], optional): Grid size which have been applied. Default (7, 7)
    Returns:
        output (torch.Tensor): Folded output tensor of the shape [B, C, original_size[0], original_size[1]].
    """
    # Get height, width, and channels
    (H, W), C = original_size, grid.shape[-1]
    # Compute original batch size
    B = int(grid.shape[0] / (H * W / grid_size[0] / grid_size[1]))
    # Fold grid tensor
    output = grid.view(B, H // grid_size[0], W // grid_size[1], grid_size[0], grid_size[1], C)
    output = output.permute(0, 5, 3, 1, 4, 2).contiguous().view(B, C, H, W)
    return output


def get_relative_position_index(
        win_h: int,
        win_w: int
) -> torch.Tensor:
    """ Function to generate pair-wise relative position index for each token inside the window.
        Taken from Timms Swin V1 implementation.
    Args:
        win_h (int): Window/Grid height.
        win_w (int): Window/Grid width.
    Returns:
        relative_coords (torch.Tensor): Pair-wise relative position indexes [height * width, height * width].
    """
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)]))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += win_h - 1
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)


class RelativeSelfAttention(nn.Module):
    """ Relative Self-Attention similar to Swin V1. Implementation inspired by Timms Swin V1 implementation.
    Args:
        in_channels (int): Number of input channels.
        num_heads (int, optional): Number of attention heads. Default 32
        grid_window_size (Tuple[int, int], optional): Grid/Window size to be utilized. Default (7, 7)
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
            self,
            in_channels: int,
            num_heads: int = 32,
            grid_window_size: Tuple[int, int] = (7, 7),
            attn_drop: float = 0.,
            drop: float = 0.
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(RelativeSelfAttention, self).__init__()
        # Save parameters
        self.in_channels: int = in_channels
        self.num_heads: int = num_heads
        self.grid_window_size: Tuple[int, int] = grid_window_size
        self.scale: float = num_heads ** -0.5
        self.attn_area: int = grid_window_size[0] * grid_window_size[1]
        # Init layers
        self.qkv_mapping = nn.Linear(in_features=in_channels, out_features=3 * in_channels, bias=True)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Linear(in_features=in_channels, out_features=in_channels, bias=True)
        self.proj_drop = nn.Dropout(p=drop)
        self.softmax = nn.Softmax(dim=-1)
        # Define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * grid_window_size[0] - 1) * (2 * grid_window_size[1] - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(grid_window_size[0],
                                                                                    grid_window_size[1]))
        # Init relative positional bias
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def _get_relative_positional_bias(
            self
    ) -> torch.Tensor:
        """ Returns the relative positional bias.
        Returns:
            relative_position_bias (torch.Tensor): Relative positional bias.
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(self.attn_area, self.attn_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def forward(
            self,
            input: torch.Tensor
    ) -> torch.Tensor:
        """ Forward pass.
        Args:
            input (torch.Tensor): Input tensor of the shape [B_, N, C].
        Returns:
            output (torch.Tensor): Output tensor of the shape [B_, N, C].
        """
        # Get shape of input
        B_, N, C = input.shape
        # Perform query key value mapping
        qkv = self.qkv_mapping(input).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # Scale query
        q = q * self.scale
        # Compute attention maps
        attn = self.softmax(q @ k.transpose(-2, -1) + self._get_relative_positional_bias())
        # Map value with attention maps
        output = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        # Perform final projection and dropout
        output = self.proj(output)
        output = self.proj_drop(output)
        return output


class MaxViTTransformerBlock(nn.Module):
    """ MaxViT Transformer block.
        With block partition:
        x ← x + Unblock(RelAttention(Block(LN(x))))
        x ← x + MLP(LN(x))
        With grid partition:
        x ← x + Ungrid(RelAttention(Grid(LN(x))))
        x ← x + MLP(LN(x))
        Layer Normalization (LN) is applied after the grid/window partition to prevent multiple reshaping operations.
        Grid/window reverse (Unblock/Ungrid) is performed on the final output for the same reason.
    Args:
        in_channels (int): Number of input channels.
        partition_function (Callable): Partition function to be utilized (grid or window partition).
        reverse_function (Callable): Reverse function to be utilized  (grid or window reverse).
        num_heads (int, optional): Number of attention heads. Default 32
        grid_window_size (Tuple[int, int], optional): Grid/Window size to be utilized. Default (7, 7)
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        drop (float, optional): Dropout ratio of output. Default: 0.0
        drop_path (float, optional): Dropout ratio of path. Default: 0.0
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 4.0
        act_layer (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
        norm_layer (Type[nn.Module], optional): Type of normalization layer to be utilized. Default: nn.BatchNorm2d
    """

    def __init__(
            self,
            in_channels: int,
            partition_function: Callable,
            reverse_function: Callable,
            num_heads: int = 32,
            grid_window_size: Tuple[int, int] = (7, 7),
            attn_drop: float = 0.,
            drop: float = 0.,
            drop_path: float = 0.,
            mlp_ratio: float = 4.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        """ Constructor method """
        super(MaxViTTransformerBlock, self).__init__()
        # Save parameters
        self.partition_function: Callable = partition_function
        self.reverse_function: Callable = reverse_function
        self.grid_window_size: Tuple[int, int] = grid_window_size
        # Init layers
        self.norm_1 = norm_layer(in_channels)
        self.attention = RelativeSelfAttention(
            in_channels=in_channels,
            num_heads=num_heads,
            grid_window_size=grid_window_size,
            attn_drop=attn_drop,
            drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_2 = norm_layer(in_channels)
        self.mlp = Mlp(
            in_features=in_channels,
            hidden_features=int(mlp_ratio * in_channels),
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ Forward pass.
        Args:
            input (torch.Tensor): Input tensor of the shape [B, C_in, H, W].
        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C_out, H (// 2), W (// 2)].
        """
        # Save original shape
        B, C, H, W = input.shape
        # Perform partition
        input_partitioned = self.partition_function(input, self.grid_window_size)
        input_partitioned = input_partitioned.view(-1, self.grid_window_size[0] * self.grid_window_size[1], C)
        # Perform normalization, attention, and dropout
        output = input_partitioned + self.drop_path(self.attention(self.norm_1(input_partitioned)))
        # Perform normalization, MLP, and dropout
        output = output + self.drop_path(self.mlp(self.norm_2(output)))
        # Reverse partition
        output = self.reverse_function(output, (H, W), self.grid_window_size)
        return output


class MaxViTBlock(nn.Module):
    """ MaxViT block composed of MBConv block, Block Attention, and Grid Attention.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        downscale (bool, optional): If true spatial downscaling is performed. Default: False
        num_heads (int, optional): Number of attention heads. Default 32
        grid_window_size (Tuple[int, int], optional): Grid/Window size to be utilized. Default (7, 7)
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        drop (float, optional): Dropout ratio of output. Default: 0.0
        drop_path (float, optional): Dropout ratio of path. Default: 0.0
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 4.0
        act_layer (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
        norm_layer (Type[nn.Module], optional): Type of normalization layer to be utilized. Default: nn.BatchNorm2d
        norm_layer_transformer (Type[nn.Module], optional): Normalization layer in Transformer. Default: nn.LayerNorm
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            downscale: bool = False,
            num_heads: int = 32,
            grid_window_size: Tuple[int, int] = (7, 7),
            attn_drop: float = 0.,
            drop: float = 0.,
            drop_path: float = 0.,
            mlp_ratio: float = 4.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            norm_layer_transformer: Type[nn.Module] = nn.LayerNorm
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(MaxViTBlock, self).__init__()
        # Init MBConv block
        self.mb_conv = MBConv(
            in_channels=in_channels,
            out_channels=out_channels,
            downscale=downscale,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_path=drop_path
        )
        # Init Block and Grid Transformer
        self.block_transformer = MaxViTTransformerBlock(
            in_channels=out_channels,
            partition_function=window_partition,
            reverse_function=window_reverse,
            num_heads=num_heads,
            grid_window_size=grid_window_size,
            attn_drop=attn_drop,
            drop=drop,
            drop_path=drop_path,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer_transformer
        )
        self.grid_transformer = MaxViTTransformerBlock(
            in_channels=out_channels,
            partition_function=grid_partition,
            reverse_function=grid_reverse,
            num_heads=num_heads,
            grid_window_size=grid_window_size,
            attn_drop=attn_drop,
            drop=drop,
            drop_path=drop_path,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer_transformer
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ Forward pass.
        Args:
            input (torch.Tensor): Input tensor of the shape [B, C_in, H, W]
        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C_out, H // 2, W // 2] (downscaling is optional)
        """
        output = self.grid_transformer(self.block_transformer(self.mb_conv(input)))
        #  output = self.mb_conv(self.grid_transformer(input))
        return output


class MaxViTStage(nn.Module):
    """ Stage of the MaxViT.
    Args:
        depth (int): Depth of the stage.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_heads (int, optional): Number of attention heads. Default 32
        grid_window_size (Tuple[int, int], optional): Grid/Window size to be utilized. Default (7, 7)
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        drop (float, optional): Dropout ratio of output. Default: 0.0
        drop_path (float, optional): Dropout ratio of path. Default: 0.0
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 4.0
        act_layer (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
        norm_layer (Type[nn.Module], optional): Type of normalization layer to be utilized. Default: nn.BatchNorm2d
        norm_layer_transformer (Type[nn.Module], optional): Normalization layer in Transformer. Default: nn.LayerNorm
    """

    def __init__(
            self,
            depth: int,
            in_channels: int,
            out_channels: int,
            num_heads: int = 32,
            grid_window_size: Tuple[int, int] = (7, 7),
            attn_drop: float = 0.,
            drop: float = 0.,
            drop_path: Union[List[float], float] = 0.,
            mlp_ratio: float = 4.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            norm_layer_transformer: Type[nn.Module] = nn.LayerNorm
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(MaxViTStage, self).__init__()
        # Init blocks
        self.blocks = nn.Sequential(*[
            MaxViTBlock(
                in_channels=in_channels if index == 0 else out_channels,
                out_channels=out_channels,
                downscale=False,#index == 0,
                num_heads=num_heads,
                grid_window_size=grid_window_size,
                attn_drop=attn_drop,
                drop=drop,
                drop_path=drop_path if isinstance(drop_path, float) else drop_path[index],
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                norm_layer=norm_layer,
                norm_layer_transformer=norm_layer_transformer
            )
            for index in range(depth)
        ])
        self.conv = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
    def forward(self, input=torch.Tensor) -> torch.Tensor:
        """ Forward pass.
        Args:
            input (torch.Tensor): Input tensor of the shape [B, C_in, H, W].
        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C_out, H // 2, W // 2].
        """
        output = self.conv(self.blocks(input)) + input
        return output

import math
class Upsample(nn.Sequential):
    """Upsample module.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops



#Parameters
b1 = 2
b2 = 2
b3 = 4
b4 = 8
b5 = 4
b6 = 2
channel = 64
N_heads = 32
window_size = 8

# os.environ['CUDA_VISIBLE_DEVICES'] ='0'
#Maxvit ONly
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F



# Define a convolution neural network
class MAGVIT(nn.Module):
    def __init__(self, args):
        super(MAGVIT, self).__init__()
        # ((w-f+2P)/s)+1
        
        dim = channel#64
        ###############################Shallow features Extraction#####################################################
        self.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride = 1,padding='same')#3
        self.GELU_0 = nn.GELU()
        self.conv1 = nn.Conv2d(64, dim, kernel_size=3, stride = 1,padding='same')#3
        self.GELU_1 = nn.GELU()
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=3,stride = 1,padding='same')
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride = 1,padding='same')
        # self.LeakyReLU = nn.LeakyReLU(inplace=True)
        
        ###############################Deep features Extraction#####################################################
        
        self.depths = (b1,b2,b3,b4,b5, b6)
        self.downscale =False#True
        self.upscale =args.scale[0]
        self.in_channels = dim#dim
        self.out_channels = (dim, dim, dim, dim)#256
        self.num_heads = N_heads  ########### 32
        self.grid_window_size = (window_size,window_size)#(7, 7)
        self.window_size = self.grid_window_size[0]
        self.attn_drop = 0.01
        self.drop = 0.01
        self.drop_path = 0.#.1
        self.mlp_ratio = 4. #4
        self.act_layer = nn.GELU
        self.norm_layer= nn.BatchNorm2d#nn.BatchNorm2d
        self.norm_layer_transformer=nn.LayerNorm# nn.LayerNorm

        self.img_range = 1.0
        in_chans = 3
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        # self.stages = []
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        
        # for index, (depth, channel) in enumerate(zip(self.depths, self.out_channels)):
            # self.stages.append( 
            
        self.Stage_1 = MaxViTStage( 
                depth=self.depths[0],
                    in_channels=self.out_channels[0], #if index == 0 else channel,#[index - 1],
                    out_channels=self.out_channels[0],
                    num_heads=self.num_heads,
                    grid_window_size=self.grid_window_size,
                    attn_drop=self.attn_drop,
                    drop=self.drop,
                    drop_path=self.drop_path,#[sum(depths[:index]):sum(depths[:index + 1])],
                    mlp_ratio=self.mlp_ratio,
                    act_layer=self.act_layer,
                    norm_layer=self.norm_layer,
                    norm_layer_transformer=self.norm_layer_transformer
                        )
        self.conv2 = nn.Conv2d(self.out_channels[0], self.out_channels[1], kernel_size=3, stride = 1,padding='same')
        self.Stage_2 = MaxViTStage( 
                    depth=self.depths[1],
                    in_channels= self.out_channels[1],#self.in_channels if index == 0 else channel,#[index - 1],
                    out_channels=self.out_channels[1],
                    num_heads=self.num_heads,
                    grid_window_size=self.grid_window_size,
                    attn_drop=self.attn_drop,
                    drop=self.drop,
                    drop_path=self.drop_path,#[sum(depths[:index]):sum(depths[:index + 1])],
                    mlp_ratio=self.mlp_ratio,
                    act_layer=self.act_layer,
                    norm_layer=self.norm_layer,
                    norm_layer_transformer=self.norm_layer_transformer
                        )
        self.conv3 = nn.Conv2d(self.out_channels[1], self.out_channels[2], kernel_size=3, stride = 1,padding='same')
        
        self.Stage_3 = MaxViTStage( 
                    depth=self.depths[2],
                    in_channels= self.out_channels[2],#self.in_channels if index == 0 else channel,#[index - 1],
                    out_channels=self.out_channels[2],
                    num_heads=self.num_heads,
                    grid_window_size=self.grid_window_size,
                    attn_drop=self.attn_drop,
                    drop=self.drop,
                    drop_path=self.drop_path,#[sum(depths[:index]):sum(depths[:index + 1])],
                    mlp_ratio=self.mlp_ratio,
                    act_layer=self.act_layer,
                    norm_layer=self.norm_layer,
                    norm_layer_transformer=self.norm_layer_transformer
                        )
        self.conv4 = nn.Conv2d(self.out_channels[2], self.out_channels[3], kernel_size=3, stride = 1,padding='same')
        self.Stage_4 = MaxViTStage( 
                    depth=self.depths[3],
                    in_channels= self.out_channels[3], #self.out_channels[2],#self.in_channels if index == 0 else channel,#[index - 1],
                    out_channels=self.out_channels[3],#self.out_channels[3],
                    num_heads=self.num_heads,
                    grid_window_size=self.grid_window_size,
                    attn_drop=self.attn_drop,
                    drop=self.drop,
                    drop_path=self.drop_path,#[sum(depths[:index]):sum(depths[:index + 1])],
                    mlp_ratio=self.mlp_ratio,
                    act_layer=self.act_layer,
                    norm_layer=self.norm_layer,
                    norm_layer_transformer=self.norm_layer_transformer
                        )
        # self.conv3 = nn.Conv2d(dim, dim, kernel_size=3, stride = 1,padding='same')
        self.Stage_5 = MaxViTStage( 
                    depth=self.depths[3],
                    in_channels= self.out_channels[2], #self.out_channels[2],#self.in_channels if index == 0 else channel,#[index - 1],
                    out_channels=self.out_channels[3],#self.out_channels[3],
                    num_heads=self.num_heads,
                    grid_window_size=self.grid_window_size,
                    attn_drop=self.attn_drop,
                    drop=self.drop,
                    drop_path=self.drop_path,#[sum(depths[:index]):sum(depths[:index + 1])],
                    mlp_ratio=self.mlp_ratio,
                    act_layer=self.act_layer,
                    norm_layer=self.norm_layer,
                    norm_layer_transformer=self.norm_layer_transformer
                        )
            
        self.Stage_6 = MaxViTStage( 
                    depth=self.depths[3],
                    in_channels= self.out_channels[2], #self.out_channels[2],#self.in_channels if index == 0 else channel,#[index - 1],
                    out_channels=self.out_channels[3],#self.out_channels[3],
                    num_heads=self.num_heads,
                    grid_window_size=self.grid_window_size,
                    attn_drop=self.attn_drop,
                    drop=self.drop,
                    drop_path=self.drop_path,#[sum(depths[:index]):sum(depths[:index + 1])],
                    mlp_ratio=self.mlp_ratio,
                    act_layer=self.act_layer,
                    norm_layer=self.norm_layer,
                    norm_layer_transformer=self.norm_layer_transformer
                        )
        # self.conv4 = nn.Conv2d(dim, dim, kernel_size=3, stride = 1,padding='same')                # )
        self.conv_after_body = nn.Conv2d(self.out_channels[3], dim, 3, 1, 1)
        ###############################Reconstruction Image #####################################################

        
        self.conv_before_upsample = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1),
                                                      nn.GELU())#nn.LeakyReLU(inplace=True))
        self.upsample_1 = Upsample(args.scale[0], dim)
        # self.upsample_2 = Upsample(2, dim)

        self.conv_last = nn.Conv2d(dim, 3, 3, 1, 1)#3
        # self.Tanh = nn.Tanh()
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
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
    
    def forward(self, x):
        H, W = x.shape[2:]
        x_size = (x.shape[2], x.shape[3])
        x = self.check_image_size(x)
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range#  1.0
        x = self.sub_mean(x)
        ##################################################
        x = self.conv0(x)
        x = self.GELU_0(x)
        
        x = self.conv1(x)
        x = self.GELU_1(x)
        x_o = x
        ##################################################
        x = self.Stage_1(x) +x_o
        # x = self.conv2(x) 
        x = self.Stage_2(x) + x_o #+x_1
        # x = self.conv3(x) 
        x = self.Stage_3(x) + x_o #+x_2 #+x_1 
        # x = self.conv4(x) 
        x = self.Stage_4(x) + x_o #+ x_3 #+ x_1 + x_2 

        x = self.Stage_5(x) + x_o #+x_4 #+  x_1 + x_2 + x_3
        
        x = self.Stage_6(x) + x_o #+ x_5 #+ x_1 + x_2 + x_3+ x_4+ x_5
        
        x = self.conv_after_body(x) + x_o #+ x_6 #+ x_1 + x_2 + x_3+ x_4+ x_5
        ##################################################
        x = self.conv_before_upsample(x) #+ x_o
        x = self.upsample_1(x)
        # x = self.upsample_2(x)
        # x = self.upsample_3(x)
        x = self.conv_last(x)
        
        x = x / self.img_range + self.mean
        x = self.add_mean(x)
        # return x
        return x[:, :, :H*self.upscale, :W*self.upscale]
    
    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops
#         # x = x.permute(0, 2, 3, 1)
#         # x = self.VOLO_Blocks(x)
#         for block in self.network:
#             x = block(x)
#         # x = self.patch_unembed_1(x, x_size) 
        
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
#         x = self.upsample_2(x)
#         x= self.conv4(x)
# #         # x = self.last(x)
#         return x

# # # # Instantiate a neural network model 
# NetworkIR_ = NetworkIR()
# Network_IR_ = NetworkIR_.cuda() 
# summary(Network_IR_, (3, 48, 48))
def make_model(args, parent=False):
    return MAGVIT(args)
