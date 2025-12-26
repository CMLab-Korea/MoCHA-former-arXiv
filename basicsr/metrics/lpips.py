import cv2
import numpy as np
import torch
import torch.nn.functional as F
import contextlib
import io

from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.color_util import rgb2ycbcr_pt
from basicsr.utils.registry import METRIC_REGISTRY

import lpips

@METRIC_REGISTRY.register()
def calculate_lpips_pt(img, img2, net_type='alex', crop_border=0, test_y_channel=False, **kwargs):
    """Calculate LPIPS (Learned Perceptual Image Patch Similarity) (PyTorch version).

    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        net_type (str): The type of network used to calculate LPIPS. Default: 'alex' (can be 'vgg', 'squeeze').
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        Tensor: LPIPS result for each image pair.
    """

    # Ensure the images have the same shape
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    # Crop borders if needed
    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    # Convert to Y channel if specified
    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    # Ensure inputs are in the required range [0, 1] and of dtype float32
    img = img.to(torch.float32)
    img2 = img2.to(torch.float32)

    # Initialize the LPIPS model (use GPU if available)
    with contextlib.redirect_stdout(io.StringIO()):
        lpips_model = lpips.LPIPS(net=net_type).to(img.device)

    # Calculate LPIPS for each image in the batch
    lpips_score = lpips_model(img, img2, normalize=True)

    # Return the LPIPS score for each pair of images
    return lpips_score.squeeze()

# Example usage of calculate_lpips_pt function
# img1 and img2 should be in the range [0, 1] and have the shape (n, 3/1, h, w)
# lpips_score = calculate_lpips_pt(img1, img2)
