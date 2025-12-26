import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────── helper ─────────────────────── #
def depthwise_conv(c, k=3, dilation=1):
    """Depth‑wise Conv2d with ‘same’ padding."""
    pad = dilation * (k // 2)
    return nn.Conv2d(c, c, k, padding=pad, dilation=dilation,
                     groups=c, bias=True)

class SEBlock(nn.Module):
    """Squeeze‑and‑Excitation (channel attention)."""
    def __init__(self, c, r=8):
        super().__init__()
        self.fc1 = nn.Conv2d(c, c // r, 1)
        self.fc2 = nn.Conv2d(c // r, c, 1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w


# ────────────────────── Amplitude branch v2 ────────────────────── #
class AmpBranch(nn.Module):
    """
    3‑Way Branch
      • Branch‑1 (d=1) : DWConv → 1×1 Conv → DWConv
      • Branch‑2 (d=2) : DWConv → 1×1 Conv → DWConv  (dilated=2)
      • Branch‑3 (d=4) : DWConv → 1×1 Conv → DWConv  (dilated=4)
    concat → 1×1 Conv  (channel mix)  ←※ fuse 단계
    """
    def __init__(self, c):
        super().__init__()
        self.branch1 = nn.Sequential(
            depthwise_conv(c, 3, dilation=1),
            nn.GELU(),
            nn.Conv2d(c, c, 1),
            nn.GELU(),
            depthwise_conv(c, 3, dilation=1),
        )
        self.branch2 = nn.Sequential(
            depthwise_conv(c, 3, dilation=2),
            nn.GELU(),
            nn.Conv2d(c, c, 1),
            nn.GELU(),
            depthwise_conv(c, 3, dilation=2),
        )
        self.branch3 = nn.Sequential(
            depthwise_conv(c, 3, dilation=4),
            nn.GELU(),
            nn.Conv2d(c, c, 1),
            nn.GELU(),
            depthwise_conv(c, 3, dilation=4),
        )
        # 브랜치 합친 뒤 채널 재조정
        self.fuse = nn.Conv2d(3 * c, c, 1)

    def forward(self, A):
        A_log = torch.log1p(A)                 # 안정화
        f1 = self.branch1(A_log)
        f2 = self.branch2(A_log)
        f3 = self.branch3(A_log)
        A_cat = torch.cat([f1, f2, f3], dim=1) # [B, 3C, H, Wf]
        A_out = F.relu(self.fuse(A_cat), inplace=True)
        return A_out


# ─────────────────────── Phase branch ─────────────────────── #
class PhaseBranch(nn.Module):
    """
    Depth‑wise Conv → 1×1 Conv → Depth‑wise Conv → Channel Attention(SE)
    Phase 입력은 [cosΦ, sinΦ] (2C채널)
    """
    def __init__(self, c):
        super().__init__()
        ch = 2 * c
        self.dw1 = depthwise_conv(ch, 3, dilation=1)
        self.gelu = nn.GELU()
        self.pw  = nn.Conv2d(ch, ch, 1)
        self.dw2 = depthwise_conv(ch, 3, dilation=1)
        self.se  = SEBlock(ch, r=8)

    def forward(self, cos_sin, amp=None):
        # (선택) amplitude 가중 smoothing
        x = self.dw1(cos_sin)
        x = self.gelu(self.pw(x))
        x = self.dw2(x)
        x = self.se(x)
        return x




"""
RSTB 의 마지막 conv 를 대체하기 위한 block
    1. ampliutde 와 phase 를 분리
    2. amplitude 에는 아래 3개의 branch 를 적용
        * dilated depthwise conv -> 1x1 conv -> dilated depthwise conv (dilation = 0 -> general conv) 
        * dilated depthwise conv -> 1x1 conv -> dilated depthwise conv (dilation = 2)
        * dilated depthwise conv -> 1x1 conv -> dilated depthwise conv (dilation = 4)
        * 이는 amplitude 의 spatial 정보를 다양한 receptive field 로 추출하기 위함
    3. phase 에는 cos/sin 인코딩을 적용한 후 아래 2개의 모듈을 적용
        * dilated depthwise conv -> 1x1 conv -> dilated depthwise conv
        * channel attention (SEBlock) 를 적용
    4. ifft 를 수행
            
"""
# ─────────────────── Frequency‑Decoupled Block ─────────────────── #
class FreqDecoupleBlock(nn.Module):
    """
    하나의 RSTB 마지막 1×1 Conv를 대체할 수 있는 Drop‑in 모듈
    입력:  [B, C, H, W]
    출력:  [B, C, H, W]  (Residual 포함)
    """
    def __init__(self, channels=3):
        super().__init__()
        self.amp_branch   = AmpBranch(channels)
        self.phase_branch = PhaseBranch(channels)
        # 최종 residual 조정용 1×1 (선택)
        self.fuse = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        # ─ FFT & 분해 ─
        F_uv = torch.fft.rfftn(x, dim=(-2, -1), norm='ortho')      # [B,C,H,Wf]
        A    = torch.abs(F_uv)
        Phi  = torch.angle(F_uv)

        # ─ Amplitude branch ─
        A_hat = self.amp_branch(A)                                 # [B,C,H,Wf]

        # ─ Phase branch (cos/sin 인코딩) ─
        cos_sin = torch.cat([torch.cos(Phi), torch.sin(Phi)], 1)   # [B,2C,H,Wf]
        cs_hat  = self.phase_branch(cos_sin, amp=A_hat)
        cos_hat, sin_hat = cs_hat.chunk(2, dim=1)
        Phi_hat = torch.atan2(sin_hat, cos_hat)                    # [B,C,H,Wf]

        # ─ 재조합 & IFFT ─
        real = A_hat * torch.cos(Phi_hat)
        imag = A_hat * torch.sin(Phi_hat)
        F_hat = torch.complex(real, imag)
        x_hat = torch.fft.irfftn(F_hat, s=(H, W), dim=(-2, -1), norm='ortho')

        # Residual 연결
        out = x + self.fuse(x_hat)
        return out






# ───────────────────────── util ───────────────────────── #
def window_partition(x, ws):
    """
    x : [B, C, H, W]
    return : [B*nW, C, ws, ws]
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // ws, ws, W // ws, ws)  # (B, C, H/ws, ws, W/ws, ws)
    x = x.permute(0, 2, 4, 1, 3, 5)             # (B, H/ws, W/ws, C, ws, ws)
    windows = x.contiguous().view(-1, C, ws, ws)
    return windows  # B*nW, C, ws, ws


def window_reverse(windows, ws, H, W):
    """
    windows : [B*nW, C, ws, ws]
    return  : [B, C, H, W]
    """
    Bn, C, _, _ = windows.shape
    B = Bn // (H // ws * W // ws)
    x = windows.view(B, H // ws, W // ws, C, ws, ws)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    x = x.view(B, C, H, W)
    return x


# ───────────── 기존 AmpBranch / PhaseBranch 재사용 ───────────── #
# depthwise_conv, SEBlock 정의가 이미 존재한다고 가정
# (질문에서 주신 클래스 그대로 두면 됩니다)

class FreqDecoupleBlock2(nn.Module):
    """
    RSTB 마지막 1×1 Conv 대체 모듈 – window‑wise FFT + auto‑padding
    """
    def __init__(self, channels=3, window_size=8):
        super().__init__()
        self.ws = window_size
        self.amp_branch   = AmpBranch(channels)
        self.phase_branch = PhaseBranch(channels)
        self.fuse = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        """
        x : [B, C, H, W]
        """
        B, C, H, W = x.shape
        ws = self.ws

        # ───────────── 1. 패딩 계산 & 적용 ───────────── #
        pad_h = (ws - H % ws) % ws       # 0 ~ ws-1
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:               # 하나라도 0이 아니면 패딩
            x = F.pad(x, (0, pad_w, 0, pad_h))   # (left,right, top,bottom) 순서
        Hp, Wp = H + pad_h, W + pad_w    # padded 크기

        # ───────────── 2. Window Partition & FFT ───────────── #
        x_w = window_partition(x, ws)                     # [B*nW, C, ws, ws]
        F_uv = torch.fft.rfftn(x_w, dim=(-2, -1), norm='ortho')
        A, Phi = torch.abs(F_uv), torch.angle(F_uv)

        # ───────────── 3. Branch 처리 ───────────── #
        A_hat = self.amp_branch(A)
        cs_hat  = self.phase_branch(torch.cat([torch.cos(Phi), torch.sin(Phi)], 1))
        cos_hat, sin_hat = cs_hat.chunk(2, dim=1)
        Phi_hat = torch.atan2(sin_hat, cos_hat)

        # ───────────── 4. IFFT & Window Reverse ───────────── #
        real = A_hat * torch.cos(Phi_hat)
        imag = A_hat * torch.sin(Phi_hat)
        F_hat = torch.complex(real, imag)
        x_hat_w = torch.fft.irfftn(F_hat, s=(ws, ws), dim=(-2, -1), norm='ortho')
        x_hat = window_reverse(x_hat_w, ws, Hp, Wp)       # [B, C, Hp, Wp]

        # ───────────── 5. 패딩 제거 후 Residual ───────────── #
        if pad_h or pad_w:
            x_hat = x_hat[:, :, :H, :W]  
            x = x[:, :, :H, :W]# crop to original size
        out = x + self.fuse(x_hat)
        return out
    
    

# ───────────────────── Conv‑PReLU‑Conv 블록 ───────────────────── #
class CPC(nn.Module):
    """Conv → PReLU → Conv"""
    def __init__(self, channels, k=3):
        super().__init__()
        pad = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, k, padding=pad, bias=True),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, k, padding=pad, bias=True),
        )

    def forward(self, x):
        return self.block(x)


# ────────────────────────── FRB 모듈 ────────────────────────── #
class FreqDecoupleBlock3(nn.Module):
    """
    입력  : [B, C, H, W]  (실수 feature map)
    출력  : [B, C, H, W]  (Residual 연결 포함)
    """
    def __init__(self, channels: int):
        super().__init__()
        self.amp_cpc   = CPC(channels)   # Amplitude branch
        self.phase_cpc = CPC(channels)   # Phase     branch
        self.fuse      = nn.Conv2d(channels, channels, 1)  # (선택) 출력 조정

    # ────────────── Phase 보정 보조 함수 ────────────── #
    @staticmethod
    def _phase_wrap(phi):
        """‑π ~ π 범위로 wrap (numerical stability용)"""
        return (phi + torch.pi) % (2 * torch.pi) - torch.pi

    def forward(self, x):
        B, C, H, W = x.shape

        # 1) FFT (전역)
        F_uv = torch.fft.rfftn(x, dim=(-2, -1), norm='ortho')  # [B,C,H,W/2+1]
        A    = torch.abs(F_uv)
        Phi  = torch.angle(F_uv)

        # 2) Amplitude branch  (Residual: Â = A + ΔA)
        A_delta = self.amp_cpc(A)
        A_hat   = A + A_delta

        # 3) Phase branch      (Residual: P̄ = P + ΔP)
        Phi_delta = self.phase_cpc(Phi)
        Phi_hat   = self._phase_wrap(Phi + Phi_delta)

        # 4) 복합수 재조합 → IFFT
        real = A_hat * torch.cos(Phi_hat)
        imag = A_hat * torch.sin(Phi_hat)
        F_hat = torch.complex(real, imag)
        x_hat = torch.fft.irfftn(F_hat, s=(H, W), dim=(-2, -1), norm='ortho')

        # 5) Skip 연결 + (선택) 1×1 Conv
        out = x + self.fuse(x_hat)
        return out
    
    

class FrequencyRefinementBlock(nn.Module):
    """
    Phase 입력 [2C] → DW-Conv → 1x1 Conv → DW-Conv → SEBlock
    """
    def __init__(self, c, ftype=None):
        super().__init__()
        if ftype == 'amp':
            ch = c
        elif ftype == 'pha':    
            ch = 2 * c
        self.body = nn.Sequential(
            depthwise_conv(ch, 3, 1), nn.GELU(),
            nn.Conv2d(ch, ch, 1),     nn.GELU(),
            depthwise_conv(ch, 3, 1),
            SEBlock(ch)
        )
    def forward(self, cos_sin):
        return self.body(cos_sin)

# ──────────── WindowFrequencyBlock ──────────── #
class WindowFrequencyBlock(nn.Module):
    """
    RSTB 마지막 1×1 Conv 대체 모듈 – window‑wise FFT + auto‑padding
    (AmpBranch / PhaseBranch 동일 구조)
    """
    def __init__(self, channels=3, window_size=8):
        super().__init__()
        self.ws = window_size
        self.amp_branch   = FrequencyRefinementBlock(channels, ftype='amp')
        self.phase_branch = FrequencyRefinementBlock(channels, ftype='pha')
        self.fuse = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.ws

        # 1) auto‑padding
        pad_h, pad_w = (ws - H % ws) % ws, (ws - W % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w

        # 2) window FFT
        x_w  = window_partition(x, ws)                       # [B*nW, C, ws, ws]
        F_uv = torch.fft.rfftn(x_w, dim=(-2, -1), norm='ortho')
        A, Phi = torch.abs(F_uv), torch.angle(F_uv)

        # 3) branch 처리
        A_hat   = self.amp_branch(A)
        cs_hat  = self.phase_branch(torch.cat([torch.cos(Phi),
                                               torch.sin(Phi)], dim=1))
        cos_hat, sin_hat = cs_hat.chunk(2, dim=1)
        Phi_hat = torch.atan2(sin_hat, cos_hat)

        # 4) IFFT
        real, imag = A_hat * torch.cos(Phi_hat), A_hat * torch.sin(Phi_hat)
        F_hat      = torch.complex(real, imag)
        x_hat_w    = torch.fft.irfftn(F_hat, s=(ws, ws), dim=(-2, -1), norm='ortho')
        x_hat      = window_reverse(x_hat_w, ws, Hp, Wp)

        # 5) crop & residual
        if pad_h or pad_w:
            x_hat = x_hat[:, :, :H, :W]
            x     = x[:, :, :H, :W]
        return x + self.fuse(x_hat)
    
# ───────────── 기존 AmpBranch / PhaseBranch 재사용 ───────────── #
# depthwise_conv, SEBlock 정의가 이미 존재한다고 가정
# (질문에서 주신 클래스 그대로 두면 됩니다)

class FreqDecoupleBlock4(nn.Module):
    """
    RSTB 마지막 1×1 Conv 대체 모듈 – window‑wise FFT + auto‑padding
    """
    def __init__(self, channels=3, window_size=8):
        super().__init__()
        self.ws = window_size
        self.amp_branch   = AmpBranch(channels)
        self.phase_branch = PhaseBranch(channels)
        self.fuse = nn.Conv2d(channels, channels, 1)
        self.fdb_weight = nn.Parameter(torch.tensor(0.01))

    def forward(self, x):
        """
        x : [B, C, H, W]
        """
        B, C, H, W = x.shape
        ws = self.ws

        # ───────────── 1. 패딩 계산 & 적용 ───────────── #
        pad_h = (ws - H % ws) % ws       # 0 ~ ws-1
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:               # 하나라도 0이 아니면 패딩
            x = F.pad(x, (0, pad_w, 0, pad_h))   # (left,right, top,bottom) 순서
        Hp, Wp = H + pad_h, W + pad_w    # padded 크기

        # ───────────── 2. Window Partition & FFT ───────────── #
        x_w = window_partition(x, ws)                     # [B*nW, C, ws, ws]
        F_uv = torch.fft.rfftn(x_w, dim=(-2, -1), norm='ortho')
        A, Phi = torch.abs(F_uv), torch.angle(F_uv)

        # ───────────── 3. Branch 처리 ───────────── #
        A_hat = self.amp_branch(A)
        cs_hat  = self.phase_branch(torch.cat([torch.cos(Phi), torch.sin(Phi)], 1))
        cos_hat, sin_hat = cs_hat.chunk(2, dim=1)
        Phi_hat = torch.atan2(sin_hat, cos_hat)

        # ───────────── 4. IFFT & Window Reverse ───────────── #
        real = A_hat * torch.cos(Phi_hat)
        imag = A_hat * torch.sin(Phi_hat)
        F_hat = torch.complex(real, imag)
        x_hat_w = torch.fft.irfftn(F_hat, s=(ws, ws), dim=(-2, -1), norm='ortho')
        x_hat = window_reverse(x_hat_w, ws, Hp, Wp)       # [B, C, Hp, Wp]

        # ───────────── 5. 패딩 제거 후 Residual ───────────── #
        if pad_h or pad_w:
            x_hat = x_hat[:, :, :H, :W]  
            x = x[:, :, :H, :W]# crop to original size
        out = x + self.fuse(x_hat) * self.fdb_weight
        return out

# ───────────── 기존 AmpBranch / PhaseBranch 재사용 ───────────── #
# depthwise_conv, SEBlock 정의가 이미 존재한다고 가정
# (질문에서 주신 클래스 그대로 두면 됩니다)

class FreqDecoupleBlock5(nn.Module):
    """
    RSTB 마지막 1×1 Conv 대체 모듈 – window‑wise FFT + auto‑padding
    """
    def __init__(self, channels=3, window_size=8):
        super().__init__()
        self.ws = window_size
        self.amp_branch   = AmpBranch(channels)
        self.phase_branch = PhaseBranch(channels)
        self.fuse = nn.Conv2d(channels, channels, 1)
        self.fdb_weight = 0.01

    def forward(self, x):
        """
        x : [B, C, H, W]
        """
        B, C, H, W = x.shape
        ws = self.ws

        # ───────────── 1. 패딩 계산 & 적용 ───────────── #
        pad_h = (ws - H % ws) % ws       # 0 ~ ws-1
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:               # 하나라도 0이 아니면 패딩
            x = F.pad(x, (0, pad_w, 0, pad_h))   # (left,right, top,bottom) 순서
        Hp, Wp = H + pad_h, W + pad_w    # padded 크기

        # ───────────── 2. Window Partition & FFT ───────────── #
        x_w = window_partition(x, ws)                     # [B*nW, C, ws, ws]
        F_uv = torch.fft.rfftn(x_w, dim=(-2, -1), norm='ortho')
        A, Phi = torch.abs(F_uv), torch.angle(F_uv)

        # ───────────── 3. Branch 처리 ───────────── #
        A_hat = self.amp_branch(A)
        cs_hat  = self.phase_branch(torch.cat([torch.cos(Phi), torch.sin(Phi)], 1))
        cos_hat, sin_hat = cs_hat.chunk(2, dim=1)
        Phi_hat = torch.atan2(sin_hat, cos_hat)

        # ───────────── 4. IFFT & Window Reverse ───────────── #
        real = A_hat * torch.cos(Phi_hat)
        imag = A_hat * torch.sin(Phi_hat)
        F_hat = torch.complex(real, imag)
        x_hat_w = torch.fft.irfftn(F_hat, s=(ws, ws), dim=(-2, -1), norm='ortho')
        x_hat = window_reverse(x_hat_w, ws, Hp, Wp)       # [B, C, Hp, Wp]

        # ───────────── 5. 패딩 제거 후 Residual ───────────── #
        if pad_h or pad_w:
            x_hat = x_hat[:, :, :H, :W]  
            x = x[:, :, :H, :W]# crop to original size
        out = x + self.fuse(x_hat) * self.fdb_weight
        return out
    