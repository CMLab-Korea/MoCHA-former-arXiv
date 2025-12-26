import os
import torch
import torch.nn as nn
from einops import rearrange
# from pytorch_wavelets import DWTForward, DWTInverse, DTCWTForward, DTCWTInverse # (or import DWT, IDWT)
from vd.archs.base_utils_arch import Channel_Attn
# from vd.archs.swin2d_arch import SwinTransformerBlock2D
# from vd.archs.cswin_arch import DirectionalCSWinBlock
from torch.nn import functional as F
import math

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, groups=in_channels, bias=False
        )
        self.gelu = nn.GELU()
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.gelu(x)
        x = self.pointwise(x)
        return x
    
    
class DepthwiseSeparableConv_v2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv_v2, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, groups=in_channels, bias=False
        )
        self.gelu = nn.GELU()
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        # input x: (b t h w c)
        b, t, h, w, c = x.shape
        x = rearrange(x, 'b t h w c -> (b t) c h w', b=b, t=t).contiguous()
        x = self.depthwise(x)
        x = self.gelu(x)
        x = self.pointwise(x)
        x = rearrange(x, '(b t) c h w -> b t h w c', b=b, t=t).contiguous()
        return x    
    
    
class DepthwiseSeparableConv_v3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv_v3, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, groups=in_channels, bias=False
        )
        self.gelu = nn.GELU()
        self.pointwise_group = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=4, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.gelu(x)
        x = self.pointwise_group(x)
        return x

    
# Spatial-Frequency Fusion module
def SPF_ver1(spa, freq, fuse_type='add'):
    
    if fuse_type == 'add':
        out = spa + freq

    return out




class RFFTChannelAttention(nn.Module):
    r"""Fourier Channel Attention using 2‑D real FFT (rFFT).

    Args
    ----
    channels : int
        입력 feature map 채널 수.
    reduction : int, optional
        채널 attention 목으로 사용하는 bottleneck 비율. (default: 16)
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channels = channels

        # rFFT 결과(real + imag)를 다시 C개 채널로 압축
        self.conv_f = nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False)
        self.act_f  = nn.ReLU(inplace=True)

        # 기존 Squeeze‑and‑Excitation과 동일한 2‑레이어 MLP
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc1    = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.act_c  = nn.ReLU(inplace=True)
        self.fc2    = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        self.gate   = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            입력 feature map, shape = (B, C, H, W).

        Returns
        -------
        torch.Tensor
            채널 attention이 적용된 feature map, shape = (B, C, H, W).
        """
        B, C, H, W = x.shape

        # (1) 2‑D rFFT ‑‑> 복소수 결과 (H, W/2+1)
        F_r = torch.fft.rfft2(x, norm='ortho')              # complex64/128

        # (2) real/imag 분리 후 채널 차원으로 결합
        F_cat = torch.cat([F_r.real, F_r.imag], dim=1)      # shape = (B, 2C, H, W//2+1)

        # (3) 1×1 Conv → ReLU (frequency‑domain projection)
        F_proj = self.act_f(self.conv_f(F_cat))             # shape = (B, C, H, W//2+1)

        # (4) Global Average Pool → 두 단계 FC → Sigmoid
        w = self.pool(F_proj)                               # (B, C, 1, 1)
        w = self.act_c(self.fc1(w))
        w = self.gate(self.fc2(w))                          # (B, C, 1, 1) ∈ (0,1)

        # (5) 원래 spatial feature에 채널‑attention 적용
        out = x * w                                         # broadcasting
        return out
    
# ----------------------------------------------------------------------
# 기본 유틸
# ----------------------------------------------------------------------
def depthwise_conv(in_channels: int,
                   kernel_size: int = 3,
                   stride: int = 1,
                   bias: bool = True) -> nn.Conv2d:
    """3×3 Depthwise Conv 래퍼"""
    padding = kernel_size // 2
    return nn.Conv2d(in_channels,
                     in_channels,
                     kernel_size,
                     stride,
                     padding,
                     groups=in_channels,
                     bias=bias)

# ----------------------------------------------------------------------
# FSM Module (Conv 스택 교체 버전)
# ----------------------------------------------------------------------
class FSM(nn.Module):
    r"""
    Fourier Split Module
    ├─ FFT
    ├─ Amplitude  branch : ①③⑤⑦ (3×3 → 3×3 DW → 1×1 → 3×3 DW)
    ├─ Phase      branch : ②④⑥   (1×1 → 3×3 DW → 1×1)
    ├─ Feature Fusion + Channel‑wise Scaling
    └─ IFFT
    """
    def __init__(self, channels: int = 64, act_layer=nn.GELU):
        super().__init__()
        self.channels = channels
        self.act = act_layer()

        # ------------------- Amplitude branch --------------------------
        self.amp_conv = nn.Sequential(
            depthwise_conv(channels),                      # (③) 3×3 DWConv
            self.act,
            nn.Conv2d(channels, channels, 1),              # (⑤) 1×1 Conv
            self.act,
            depthwise_conv(channels),                      # (⑦) 3×3 DWConv
        )

        # ------------------- Phase branch -----------------------------
        self.phase_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1),              # (②) 1×1 Conv
            self.act,
            depthwise_conv(channels),                      # (④) 3×3 DWConv
            self.act,
            nn.Conv2d(channels, channels, 1),              # (⑥) 1×1 Conv
        )

        # ---------------- Feature fusion & scaling --------------------
        # 두 분기를 concat 후 3×3 → 3×3
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 3, padding=1),
            self.act,
            nn.Conv2d(channels, channels, 3, padding=1),
            self.act,
        )
        # Global Average Pooling → 채널 스케일링 weight (Amplitude/Phase 각각)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.scale_conv = nn.Conv2d(channels, 2 * channels, 1)  # 2C (= amp, phase)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x : (B, C, H, W)   – real‑valued input image/feature
        """
        B, C, H, W = x.shape
        assert C == self.channels, "채널 수가 설정과 다릅니다."

        # 1) Fourier transform
        freq = torch.fft.rfft2(x, norm='ortho')            # (B, C, H, W/2+1), complex
        amp  = torch.abs(freq)                             # Amplitude  (real  tensor)
        phase= torch.angle(freq)                           # Phase      (real  tensor)

        # 2) 각각 Conv 스택
        amp_feat   = self.amp_conv(amp)
        phase_feat = self.phase_conv(phase)

        # 3) Feature fusion
        fuse = self.fusion(torch.cat([amp_feat, phase_feat], dim=1))

        # 4) 채널 스케일링(σ)
        scale = torch.sigmoid(self.scale_conv(self.gap(fuse)))  # (B, 2C, 1, 1)
        amp_scale, phase_scale = torch.chunk(scale, 2, dim=1)

        # 5) 스케일 적용
        amp_mod   = amp_feat   * amp_scale
        phase_mod = phase_feat * phase_scale

        # 6) 복합수 재구성 후 IFFT
        #    복합수:  r = amp · exp(j·phase)
        complex_r = amp_mod * torch.exp(1j * phase_mod)
        out = torch.fft.irfft2(complex_r, s=(H, W), norm='ortho')
        
        out = out + x # residual connection

        return out
    
    
class FourierUnit(nn.Module):
    def __init__(self, embed_dim, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.conv_layer = torch.nn.Conv2d(embed_dim * 2, embed_dim * 2, 1, 1, 0)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(ffted)

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4,
                                                                       2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        return output

# 아래는 내가 수정한 코드    
class FourierUnit2(nn.Module):
    def __init__(self, embed_dim, fft_norm='ortho', amp_relu=True):
        super().__init__()
        self.fft_norm = fft_norm
        self.amp_relu = amp_relu             # amplitude ≥0 보장 여부

        # ───── 기존 실·허수 branch ───── #
        self.conv_layer = nn.Conv2d(embed_dim * 2, embed_dim * 2, 1, 1, 0)
        self.relu       = nn.LeakyReLU(0.2, inplace=True)

        # ───── 새 Amplitude branch ───── #
        self.amp_conv1 = nn.Conv2d(embed_dim, embed_dim, 1, 1, 0)
        self.amp_act   = nn.LeakyReLU(0.2, inplace=True)
        self.amp_conv2 = nn.Conv2d(embed_dim, embed_dim, 1, 1, 0)

        # ───── 새 Phase branch ───── #
        self.ph_conv1 = nn.Conv2d(embed_dim, embed_dim, 1, 1, 0)
        self.ph_act   = nn.LeakyReLU(0.2, inplace=True)
        self.ph_conv2 = nn.Conv2d(embed_dim, embed_dim, 1, 1, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        fft_dim = (-2, -1)

        # ===== rFFT + Real/Imag Conv (원본 코드) =====
        f = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)          # complex (B,C,H,W//2+1)
        f = torch.stack((f.real, f.imag), dim=-1)                        # (B,C,H,Wf,2)
        f = f.permute(0, 1, 4, 2, 3).contiguous()                        # (B,C,2,H,Wf)
        f = f.view(B, -1, H, f.shape[-1])                                # (B,C*2,H,Wf)

        f = self.relu(self.conv_layer(f))                                # (B,C*2,H,Wf)

        # 복소 복원
        f = f.view(B, C, 2, H, f.shape[-1]).permute(0,1,3,4,2).contiguous()  # (B,C,H,Wf,2)
        f = torch.complex(f[...,0], f[...,1])                            # complex (B,C,H,Wf)

        # ===== 여기서부터 Amplitude / Phase branch =====
        amp   = torch.abs(f)                                             # amplitude  (B,C,H,Wf)
        phase = torch.angle(f)                                           # phase(rad) (B,C,H,Wf)

        # --- amplitude branch ---
        amp = self.amp_conv2(self.amp_act(self.amp_conv1(amp)))
        if self.amp_relu:
            amp = F.relu(amp)   # 음수 제거 (권장)

        # --- phase branch ---
        phase = self.ph_conv2(self.ph_act(self.ph_conv1(phase)))
        # (선택) wrap: phase = torch.atan2(torch.sin(phase), torch.cos(phase))

        # --- 재합성 ---
        f = torch.polar(amp, phase)                                      # complex (B,C,H,Wf)

        # ===== iFFT (원본) =====
        output = torch.fft.irfftn(f, s=(H, W), dim=fft_dim, norm=self.fft_norm)
        return output    

class SpectralTransform(nn.Module):
    def __init__(self, embed_dim, last_conv=False):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.last_conv = last_conv

        self.conv1 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.fu = FourierUnit2(embed_dim // 2)

        self.conv2 = torch.nn.Conv2d(embed_dim // 2, embed_dim, 1, 1, 0)

        if self.last_conv:
            self.last_conv = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        output = self.fu(x)
        output = self.conv2(x + output)
        if self.last_conv:
            output = self.last_conv(output)
        return output

## Residual Block (RB)
class ResB(nn.Module):
    def __init__(self, embed_dim, red=1):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // red, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(embed_dim // red, embed_dim, 3, 1, 1),
        )

    def __call__(self, x):
        out = self.body(x)
        return out + x

class SFB(nn.Module):
    def __init__(self, embed_dim, red=1):
        super(SFB, self).__init__()
        self.S = ResB(embed_dim, red)
        self.F = SpectralTransform(embed_dim)
        self.fusion = nn.Conv2d(embed_dim * 2, embed_dim, 1, 1, 0)

    def __call__(self, x):
        s = self.S(x)
        f = self.F(x)
        out = torch.cat([s, f], dim=1)
        out = self.fusion(out)
        return out
