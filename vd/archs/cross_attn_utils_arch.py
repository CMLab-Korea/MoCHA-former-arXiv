import sys 
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers

from einops import rearrange

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias



class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

# Cross-Attention module
#   Q: coarse clean features
#   KV: predicted moire pattern
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.dim = dim
        self.softmax = nn.Softmax(dim=-1)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def _forward(self, q, kv):
        k,v = kv.chunk(2, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax(attn)
        out = (attn @ v)
        return out

    def forward(self, low, high):
        self.h, self.w = low.shape[2:]

        q = self.q_dwconv(self.q(high))
        kv = self.kv_dwconv(self.kv(low))
        out = self._forward(q, kv)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=kv.shape[-2], w=kv.shape[-1])
        out = self.project_out(out)
        return out
    
## Improved feed-forward network
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)
        self.hid_fea = hidden_features
        self.dim = dim

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        self.h, self.w = x.shape[2:]
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
    
# conv block -> 1x1 conv + 3x3 conv
# swin block -> LayerNorm + (1x1 conv + 3x3 conv)
# low(kv): conv / high(q): conv
class FCAM_ver1(nn.Module):
    r""" Feature Cross Attention Module (clean feature + moire feature)
    This module is used to fuse the clean feature and moire feature.
    (for pay attention to the moire patterns)
    
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        ffn_expansion_factor (int): Define the window size.
        bias (int): Shift size for SW-MSA.
        LayerNorm_type (float): Ratio of mlp hidden dim to embedding dim.
    """
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(FCAM_ver1, self).__init__()

        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.dim = dim

    def forward(self, low, high):
        self.h, self.w = low.shape[2:]
        
        # FCAM 는 feature 를 enhance 하는데에 초점을 두고 있지 않음
        # FCAM 의 목적은 moire pattern 에 큰 수치를 부여하는 attention map 을 생성하는 것
        # 따라서 기존 코드와는 다르게 self.attn 의 결과에 skip connection 을 두지 않음
        # TODO: skip connection 을 두는 것과 두지 않는 것 중 어느것이 성능이 더 좋은지 확인해야 함.
        x = self.attn(low, high)
        x = self.ffn(self.norm2(x))
        
        # x = low + self.attn(self.norm1(low), high)
        # x = x + self.ffn(self.norm2(x))

        return x
    
# conv block -> 1x1 conv + 3x3 conv
# swin block -> LayerNorm + (1x1 conv + 3x3 conv)
# low(kv): conv / high(q): swin
class FCAM_ver2(nn.Module):
    r""" Feature Cross Attention Module (clean feature + moire feature)
    This module is used to fuse the clean feature and moire feature.
    (for pay attention to the moire patterns)
    
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        ffn_expansion_factor (int): Define the window size.
        bias (int): Shift size for SW-MSA.
        LayerNorm_type (float): Ratio of mlp hidden dim to embedding dim.
    """
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(FCAM_ver2, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.dim = dim

    def forward(self, low, high):
        self.h, self.w = low.shape[2:]
        
        # FCAM 는 feature 를 enhance 하는데에 초점을 두고 있지 않음
        # FCAM 의 목적은 moire pattern 에 큰 수치를 부여하는 attention map 을 생성하는 것
        # 따라서 기존 코드와는 다르게 self.attn 의 결과에 skip connection 을 두지 않음
        # TODO: skip connection 을 두는 것과 두지 않는 것 중 어느것이 성능이 더 좋은지 확인해야 함.
        x = self.attn(low, self.norm1(high))
        x = self.ffn(self.norm2(x))
        
        # x = low + self.attn(self.norm1(low), high)
        # x = x + self.ffn(self.norm2(x))

        return x
    
class M3CA(nn.Module):
    r""" M3C module
        Moire Mask Module with Cross Attention
    
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        ffn_expansion_factor (int): Define the window size.
        bias (int): Shift size for SW-MSA.
        LayerNorm_type (float): Ratio of mlp hidden dim to embedding dim.
    """
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(M3CA, self).__init__()
        self.fcam = FCAM(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
        self.norm = LayerNorm(dim, LayerNorm_type)
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)