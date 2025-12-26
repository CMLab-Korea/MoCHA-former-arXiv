import math
import numpy as np

import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY
from basicsr.losses.basic_loss import _reduction_modes
from basicsr.losses.loss_util import weighted_loss


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def l2_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@LOSS_REGISTRY.register()
class L1_Loss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1_Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class L2_Loss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L2_Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l2_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class WNNMLoss(nn.Module):
    """
    Weighted Nuclear Norm Minimization Loss.
    Assigns lower loss to images with simpler structures (higher low-rankness)
    and higher loss to complex images.
    """
    
    def __init__(self, patch_size=8, stride=4, num_iterations=3, C=2.8, loss_weight=1.0):
        """
        Args:
            patch_size: Size of the patches to extract from the image.
            stride: Stride for patch extraction.
            num_iterations: Number of iterations for WNNM.
            C: Constant for weight calculation.
        """
        super(WNNMLoss, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.num_iterations = num_iterations
        self.C = C
        self.loss_weight = loss_weight
        
    def extract_patches(self, image):
        """
        Extract patches from the image.
        
        Args:
            image: Input image of shape [B, C, H, W]
            
        Returns:
            Patches of shape [B*num_patches, C*patch_size*patch_size]
        """
        B, C, H, W = image.shape
        
        # Extract patches
        patches = F.unfold(image, kernel_size=self.patch_size, stride=self.stride)
        # Reshape to [B, C*patch_size*patch_size, num_patches]
        patches = patches.permute(0, 2, 1)
        # Reshape to [B*num_patches, C*patch_size*patch_size]
        num_patches = patches.shape[1]
        patches = patches.reshape(B * num_patches, C * self.patch_size * self.patch_size)
        
        return patches, num_patches
    
    def soft_threshold(self, singular_values, weights):
        """
        Apply soft thresholding to singular values.
        Args:
            singular_values: Singular values.
            weights: Weights for soft thresholding.
        Returns:
            Soft thresholded singular values.
        """
        return torch.sign(singular_values) * torch.maximum(torch.abs(singular_values) - weights, torch.zeros_like(singular_values))
    
    def compute_wnnm(self, patches, patch_mean, noise_sigma=0.1):
        """
        Compute WNNM for each patch.
        Args:
            patches: Image patches [num_patches, patch_dim]
            patch_mean: Mean of patches [num_patches, patch_dim]
            noise_sigma: Estimated noise standard deviation            
        Returns:
            Denoised patches and nuclear norm loss
        """
        # Center the patches
        centered_patches = patches - patch_mean
        
        # SVD
        U, S, V = torch.svd(centered_patches)
        patch_num = centered_patches.shape[0]

        # Initial temp calculation
        patch_dim = patches.shape[1]       # = C * patch_size²
        temp = torch.sqrt(torch.maximum(S ** 2 - patch_dim * (noise_sigma ** 2),
                                torch.zeros_like(S)))
        # temp = torch.sqrt(torch.maximum(S ** 2 - patch_num * (noise_sigma ** 2), torch.zeros_like(S)))
        
        # WNNM iterations
        for _ in range(self.num_iterations):
            # Calculate weight vector
            W_vec = (self.C * torch.sqrt(torch.tensor(patch_num, device=patches.device)) * (noise_sigma ** 2)) / (temp + 1e-8)
            # Apply soft thresholding
            sigma_x = self.soft_threshold(S, W_vec)
            # Update temp
            temp = sigma_x
        
        # Calculate nuclear norm (sum of singular values)
        nuclear_norm = sigma_x.sum()
        # Reconstruct patches
        reconstructed = torch.matmul(torch.matmul(U, torch.diag_embed(sigma_x)), V.transpose(-2, -1)) + patch_mean        
        return reconstructed, nuclear_norm

    def forward(self, images, noise_sigma=0.1):
        """
        Forward pass.
        Args:
            images: Input images of shape [B, C, H, W]
            noise_sigma: Estimated noise standard deviation
        Returns:
            Loss value based on low-rankness
        """
        batch_size = images.shape[0]
        # Extract patches
        patches, num_patches_per_image = self.extract_patches(images)
        # Calculate patch means
        patch_mean = patches.mean(dim=1, keepdim=True)
        # Apply WNNM to patches
        _, nuclear_norm = self.compute_wnnm(patches, patch_mean, noise_sigma)
        # Normalize the loss by the number of patches
        nuclear_norm_loss = nuclear_norm / batch_size
        # nuclear_norm_loss = nuclear_norm / (batch_size * num_patches_per_image)
        return nuclear_norm_loss * self.loss_weight


@LOSS_REGISTRY.register()
class WNNMLoss2(nn.Module):
    """
    Weighted Nuclear Norm Minimization Loss.
    Assigns lower loss to images with simpler structures (higher low-rankness)
    and higher loss to complex images.
    """
    
    def __init__(self, patch_size=8, stride=4, num_iterations=3, C=2.8, loss_weight=1.0):
        """
        Args:
            patch_size: Size of the patches to extract from the image.
            stride: Stride for patch extraction.
            num_iterations: Number of iterations for WNNM.
            C: Constant for weight calculation.
        """
        super(WNNMLoss2, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.num_iterations = num_iterations
        self.C = C
        self.loss_weight = loss_weight
        
    def extract_patches(self, image):
        """
        Extract patches from the image.
        
        Args:
            image: Input image of shape [B, C, H, W]
            
        Returns:
            Patches of shape [B*num_patches, C*patch_size*patch_size]
        """
        B, C, H, W = image.shape
        
        # Extract patches
        patches = F.unfold(image, kernel_size=self.patch_size, stride=self.stride)
        # Reshape to [B, C*patch_size*patch_size, num_patches]
        patches = patches.permute(0, 2, 1)
        # Reshape to [B*num_patches, C*patch_size*patch_size]
        num_patches = patches.shape[1]
        patches = patches.reshape(B * num_patches, C * self.patch_size * self.patch_size)
        
        idx_groups = [torch.arange(B * num_patches, device=image.device)]
        return patches, num_patches, idx_groups
    
    def soft_threshold(self, singular_values, weights):
        """
        Apply soft thresholding to singular values.
        Args:
            singular_values: Singular values.
            weights: Weights for soft thresholding.
        Returns:
            Soft thresholded singular values.
        """
        return torch.sign(singular_values) * torch.maximum(torch.abs(singular_values) - weights, torch.zeros_like(singular_values))
    
    def compute_wnnm(self, patches, patch_mean, noise_sigma=0.1):
        """
        Compute WNNM for each patch.
        Args:
            patches: Image patches [num_patches, patch_dim]
            patch_mean: Mean of patches [num_patches, patch_dim]
            noise_sigma: Estimated noise standard deviation            
        Returns:
            Denoised patches and nuclear norm loss
        """
        # Center the patches
        centered_patches = patches - patch_mean
        
        # SVD
        # U, S, V = torch.svd(centered_patches)
        U, S, Vh = torch.linalg.svd(centered_patches, full_matrices=False)  # new
        V = Vh.transpose(-2, -1)
        patch_num = centered_patches.shape[0]

        # Initial temp calculation
        patch_dim = patches.shape[1]       # = C * patch_size²
        temp = torch.sqrt(torch.maximum(S ** 2 - patch_dim * (noise_sigma ** 2),
                                torch.zeros_like(S)))
        # temp = torch.sqrt(torch.maximum(S ** 2 - patch_num * (noise_sigma ** 2), torch.zeros_like(S)))
        
        # WNNM iterations
        for _ in range(self.num_iterations):
            # Calculate weight vector
            W_vec = (self.C * torch.sqrt(torch.tensor(patch_num, device=patches.device)) * (noise_sigma ** 2)) / (temp + 1e-8)
            # Apply soft thresholding
            sigma_x = self.soft_threshold(S, W_vec)
            # Update temp
            temp = sigma_x
        
        # ① 가중치 W_vec 적용
        weighted_nuclear_norm = (W_vec * sigma_x).sum()
        # ② 에너지 정규화(선택): Frobenius norm으로 스케일 분리
        frob_norm = torch.norm(centered_patches) + 1e-8
        nuclear_norm = weighted_nuclear_norm / frob_norm
        
        # Reconstruct patches
        reconstructed = torch.matmul(torch.matmul(U, torch.diag_embed(sigma_x)), V.transpose(-2, -1)) + patch_mean        
        return reconstructed, nuclear_norm

    def forward(self, images, noise_sigma=0.1):
        """
        Forward pass.
        Args:
            images: Input images of shape [B, C, H, W]
            noise_sigma: Estimated noise standard deviation
        Returns:
            Loss value based on low-rankness
        """
        batch_size = images.shape[0]
        # Extract patches
        patches, _, idx_groups = self.extract_patches(images)
        loss_accum = 0.0
        for idx in idx_groups:
            grp = patches[idx]                       # [K, P]
            patch_mean = grp.mean(dim=1, keepdim=True).expand(-1, grp.shape[1])
            _, nuc = self.compute_wnnm(grp, patch_mean, noise_sigma)
            loss_accum += nuc
        # 이미지 개수와 그룹 개수로 평균
        nuclear_norm_loss = loss_accum / (batch_size * len(idx_groups))
        return nuclear_norm_loss * self.loss_weight
    

@LOSS_REGISTRY.register()
class OrientationEntropyLoss(nn.Module):
    r"""Orientation‑Entropy(Anisotropy) Loss for multi‑channel moiré prediction.

    Args:
        n_bins (int): Angular histogram bin 수 (기본 36 → 10° 간격).
        r_min, r_max (float): Radial mask 범위(0–1). 저주파·초고주파 제거용.
        loss_weight (float): 전체 loss에 곱해질 가중치.
        reduction (str): 'mean' | 'sum' – 배치 차원 집계 방식.
    """
    def __init__(self,
                 n_bins: int = 36,
                 r_min: float = 0.0,
                 r_max: float = 1.0,
                 loss_weight: float = 1.0,
                 reduction: str = "mean"):
        super().__init__()
        assert 0.0 <= r_min < r_max <= 1.0
        assert reduction in ("mean", "sum")
        self.n_bins = n_bins
        self.r_min = r_min
        self.r_max = r_max
        self.loss_weight = loss_weight
        self.reduction = reduction

    # ───────────────────────── internal helper ───────────────────────── #
    @staticmethod
    def _soft_histogram(bin_idx, amp, n_bins):
        floor = bin_idx.floor()
        ceil  = (floor + 1) % n_bins
        w     = bin_idx - floor
        hist = torch.zeros(n_bins,
                           device=amp.device,
                           dtype=amp.dtype)
        hist.scatter_add_(0, floor.long().view(-1),
                          amp.view(-1) * (1.0 - w).view(-1))
        hist.scatter_add_(0, ceil.long().view(-1),
                          amp.view(-1) * w.view(-1))
        return hist

    # ──────────────────────────── forward ────────────────────────────── #
    def forward(self, moire_pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            moire_pred: (B, C, H, W) – C=3(R,G,B) 예상
        Returns:
            scalar loss (또는 B* C 길이 벡터 → reduction 적용)
        """
        assert moire_pred.dim() == 4, "input must be (B,C,H,W)"
        B, C, H, W = moire_pred.shape
        device      = moire_pred.device

        # ── FFT amplitude per channel ────────────────────────────────── #
        f   = torch.fft.fftshift(torch.fft.fft2(moire_pred,
                                               norm='ortho',
                                               dim=(-2, -1)),
                                 dim=(-2, -1))            # (B,C,H,W)
        amp = torch.abs(f)                                # (B,C,H,W)
        amp = amp.reshape(B * C, H, W)                    # (B*C,H,W)

        # ── polar coordinates (공통 템플릿, 한 번만 생성) ─────────────── #
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=device),
            torch.linspace(-1.0, 1.0, W, device=device),
            indexing="ij"
        )
        rho   = torch.sqrt(xx ** 2 + yy ** 2)
        rho   = rho / rho.max()
        theta = torch.atan2(yy, xx)                       # (-π,π]
        theta = (theta + math.pi) / (2 * math.pi)         # [0,1)

        mask   = (rho >= self.r_min) & (rho <= self.r_max)
        bin_idx = theta * self.n_bins                     # (H,W)

        # ── histogram & entropy for each (batch,channel) ─────────────── #
        entropies = []
        for k in range(B * C):
            amp_k = amp[k] * mask
            hist  = self._soft_histogram(bin_idx, amp_k, self.n_bins)
            p     = hist / (hist.sum() + 1e-8)
            ent   = -(p * (p + 1e-8).log()).sum()
            entropies.append(ent)

        entropies = torch.stack(entropies)                # (B*C,)

        # ── 배치 집계(reduction) ─────────────────────────────────────── #
        if self.reduction == "mean":
            loss = entropies.mean()
        else:
            loss = entropies.sum()

        return loss * self.loss_weight

@LOSS_REGISTRY.register()
class FFTLoss(nn.Module):
    """L1 loss in frequency domain with FFT.

    Args:
        loss_weight (float): Loss weight for FFT loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FFTLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (..., C, H, W). Predicted tensor.
            target (Tensor): of shape (..., C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (..., C, H, W). Element-wise
                weights. Default: None.
        """

        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)
        return self.loss_weight * l1_loss(pred_fft, target_fft, weight, reduction=self.reduction)