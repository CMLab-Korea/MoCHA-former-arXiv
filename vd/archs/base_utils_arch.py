import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNextBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNextBlock_wGConv(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, groups=4):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1, groups=groups)  # pointwise/1x1 convs/groups=4
        # self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, 4, kernel_size=1, groups=groups)  # pointwise/1x1 convs/groups=4
        # self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
    
class ConvNextBlock3D(nn.Module):
    r"""
    ConvNeXt Block (3D 확장)  
    DwConv3d → channels_last LayerNorm → 1×1 Linear → GELU → 1×1 Linear → residual + DropPath

    Args:
        dim (int): channel 수.
        drop_path (float): stochastic depth 확률.
        layer_scale_init_value (float): Layer Scale 초기값.
        kernel_size (tuple[int,int,int]): (kT,kH,kW); 기본 (3,7,7).
    """
    def __init__(
        self,
        dim=96,
        drop_path=False,
        layer_scale_init_value=1e-6,
        kernel_size=(3, 7, 7),
    ):
        super().__init__()
        t, h, w = kernel_size
        pad = (t // 2, h // 2, w // 2)
        self.dwconv = nn.Conv3d(
            dim, dim,
            kernel_size=kernel_size,
            padding=pad,
            groups=dim          # depthwise
        )
        self.norm = LayerNorm(dim, eps=1e-6)          # channels_last 전용
        self.pwconv1 = nn.Linear(dim, 4 * dim)        # 1×1×1 pointwise (Linear)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim))
            if layer_scale_init_value > 0 else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, C, T, H, W)
        residual = x

        # (1) depthwise 3D conv
        x = self.dwconv(x)                           # (B, C, T, H, W)

        # (2) channels_last로 변환
        x = x.permute(0, 2, 3, 4, 1)                # (B, T, H, W, C)

        # (3) LayerNorm + PW-FFN
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x

        # (4) 원래 순서로 복원
        x = x.permute(0, 4, 1, 2, 3)                # (B, C, T, H, W)

        # (5) residual & stochastic depth
        x = residual + self.drop_path(x)
        return x

class Channel_Attn(nn.Module):
    def __init__(self, channel, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction_ratio, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Scale
        return x * y.expand_as(x)
    
class Spatial_Module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Spatial_Module, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class MoireGenerator(nn.Module):
    def __init__(self, in_ch, convnext_block):
        super(MoireGenerator, self).__init__()
        self.moire_generator = nn.ModuleList([ConvNextBlock(dim=in_ch) for _ in range(convnext_block)])

    def forward(self, x):
        for module in self.moire_generator:
            x = module(x)
        return x
    
    
class ConvNeXtBlocks(nn.Module):
    def __init__(self, in_ch, convnext_block):
        super(ConvNeXtBlocks, self).__init__()
        self.moire_generator = nn.ModuleList([ConvNextBlock(dim=in_ch) for _ in range(convnext_block)])

    def forward(self, x):
        for module in self.moire_generator:
            x = module(x)
        return x
    
class ConvNeXtBlocks3D(nn.Module):
    def __init__(self, 
                 in_ch=96, 
                 convnext_block=1,
                 kernel_size=(1,7,7)):
        super(ConvNeXtBlocks3D, self).__init__()
        self.moire_generator = nn.ModuleList([ConvNextBlock3D(dim=in_ch,
                                                              drop_path=False,
                                                              layer_scale_init_value=1e-6,
                                                              kernel_size=kernel_size)
                                              for _ in range(convnext_block)])

    def forward(self, x):
        for module in self.moire_generator:
            x = module(x)
        return x
    
class ConvNeXtBlocks_GConv(nn.Module):
    def __init__(self, in_ch, convnext_block,groups=4):
        super(ConvNeXtBlocks_GConv, self).__init__()
        self.moire_generator = nn.ModuleList([ConvNextBlock_wGConv(dim=in_ch, groups=groups) for _ in range(convnext_block)])

    def forward(self, x):
        for module in self.moire_generator:
            x = module(x)
        return x
    
    
class DepthwiseSeparableConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        
        # 1. Depthwise Convolution: 각 채널별로 독립적인 3D 컨볼루션 적용
        self.depthwise = nn.Conv3d(
            in_channels, 
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(0, 1, 1),  # 시간 차원은 0, 높이와 너비 차원은 1
            groups=in_channels
        )
                
        # 2. Pointwise Convolution: 1x1x1 컨볼루션으로 채널 간 정보 혼합
        self.pointwise = nn.Conv3d(
            in_channels, 
            out_channels, 
            kernel_size=(1, 1, 1)
        )
    
    def forward(self, x):
        # x: (batch_size, channels, time, height, width)
        x = self.pointwise(x)
        x = self.depthwise(x)
        return x