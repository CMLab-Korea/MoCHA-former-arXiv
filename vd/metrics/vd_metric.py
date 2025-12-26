import cv2
import numpy as np
import lpips
from basicsr.utils.registry import METRIC_REGISTRY
# import lpips
from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.color_util import rgb2ycbcr_pt

import lpips
import torch
import contextlib
import io


@METRIC_REGISTRY.register()
def calculate_vd_psnr(img, img2):
    # normalized_psnr = -10 * np.log10(np.mean(np.power(img - img2, 2)))
    img_np = (img * 255.0).round()
    img2_np = (img2 * 255.0).round()
    mse = np.mean((img_np - img2_np) ** 2)
    if mse == 0:
        return float('inf')
    # if normalized_psnr == 0:
    #     return float('inf')
    # return normalized_psnr
    return 20 * np.log10(255.0 / np.sqrt(mse))


@METRIC_REGISTRY.register()
def calculate_vd_ssim(img, img2):
    img_np = (img * 255.0).round()
    img2_np = (img2 * 255.0).round()
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img_np.astype(np.float64)
    img2 = img2_np.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

    # vd_ssim = compare_ssim(img, img2, multichannel=True)
    # return vd_ssim


@METRIC_REGISTRY.register()
def calculate_vd_lpips(img, img2, net_type='alex', test_y_channel=False):
    """Calculate LPIPS (Learned Perceptual Image Patch Similarity) for single NumPy images.

    Args:
        img_np (ndarray): Image with range [0, 1], shape (H, W, C).
        img2_np (ndarray): Image with range [0, 1], shape (H, W, C).
        net_type (str): The type of network used to calculate LPIPS. Default: 'alex'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: LPIPS score for the two images.
    """

    # Ensure the images have the same shape
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    # Convert NumPy arrays to PyTorch tensors
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # Convert to (1, C, H, W)
    img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0)  # Convert to (1, C, H, W)

    # Convert to float32 and move to the appropriate device
    img = img.to(torch.float32)
    img2 = img2.to(torch.float32)

    # Convert to Y channel if specified
    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    # Initialize the LPIPS model
    lpips_model = lpips.LPIPS(net=net_type, verbose=False).to(img.device)

    # Calculate LPIPS
    lpips_score = lpips_model(img, img2, normalize=True)

    # Return the LPIPS score as a float
    return lpips_score.item()