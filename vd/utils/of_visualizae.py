import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import flow_vis_torch
import flow_vis


def concatenate_images(image_t_minus_1_np, image_t_np, image_t_plus_1_np):
    """
    Convert numpy images to PyTorch tensors and concatenate them into a tensor of shape (3, 3, H, W).
    
    Parameters:
    - image_t_minus_1_np: Numpy array for image t-1 (HxWx3)
    - image_t_np: Numpy array for image t (HxWx3)
    - image_t_plus_1_np: Numpy array for image t+1 (HxWx3)
    
    Returns:
    - PyTorch tensor of shape (3, 3, H, W)
    """
    # Convert numpy arrays (HxWx3) to PyTorch tensors (3xHxW)
    image_t_minus_1 = torch.from_numpy(np.transpose(image_t_minus_1_np, (2, 0, 1))).float()
    image_t = torch.from_numpy(np.transpose(image_t_np, (2, 0, 1))).float()
    image_t_plus_1 = torch.from_numpy(np.transpose(image_t_plus_1_np, (2, 0, 1))).float()
    
    # Concatenate along a new dimension, resulting in shape (3, 3, H, W)
    concatenated_images = torch.stack([image_t_minus_1, image_t, image_t_plus_1], dim=0)
    
    # Add a new dimension at the beginning to get the shape (1, 3, 3, H, W)
    concatenated_images = concatenated_images.unsqueeze(0)
    
    return concatenated_images

def concatenate_flows(flow_t_to_t_minus_1_np, flow_t_to_t_plus_1_np):
    """
    Convert numpy flows to PyTorch tensors and concatenate them into a tensor of shape (2, 2, H, W).
    
    Parameters:
    - flow_t_to_t_minus_1_np: Numpy array for optical flow from t to t-1 (HxWx2)
    - flow_t_to_t_plus_1_np: Numpy array for optical flow from t to t+1 (HxWx2)
    
    Returns:
    - PyTorch tensor of shape (2, 2, H, W)
    """
    # Convert numpy arrays (HxWx2) to PyTorch tensors (2xHxW)
    flow_t_to_t_minus_1 = torch.from_numpy(np.transpose(flow_t_to_t_minus_1_np, (2, 0, 1))).float()
    flow_t_to_t_plus_1 = torch.from_numpy(np.transpose(flow_t_to_t_plus_1_np, (2, 0, 1))).float()
    
    # Concatenate along a new dimension, resulting in shape (2, 2, H, W)
    concatenated_flows = torch.stack([flow_t_to_t_minus_1, flow_t_to_t_plus_1], dim=0)
    
    # Add a new dimension at the beginning to get the shape (1, 2, 2, H, W)
    concatenated_flows = concatenated_flows.unsqueeze(0)
    
    return concatenated_flows

# RGGB 이미지를 받아서 2H x 2W x 1 그레이스케일 이미지로 변환하는 함수
def rggb_to_grayscale(rggb_image):
    
    rggb_image = rggb_image.squeeze(0)  # (1, C, H, W) -> (C, H, W)
    rggb_image = rggb_image.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
    
    rggb_image = (rggb_image - rggb_image.min()) / (rggb_image.max() - rggb_image.min()) * 255
    rggb_image = rggb_image.astype(np.uint8)
    
    H, W, _ = rggb_image.shape
    # 새로운 크기의 빈 이미지를 만듭니다.
    grayscale_image = np.zeros((2 * H, 2 * W), dtype=rggb_image.dtype)
    
    # RGGB 패턴을 각각 분해하여 배치
    grayscale_image[0::2, 0::2] = rggb_image[:, :, 0]  # R 채널
    grayscale_image[0::2, 1::2] = rggb_image[:, :, 1]  # G 채널 (첫 번째)
    grayscale_image[1::2, 0::2] = rggb_image[:, :, 2]  # G 채널 (두 번째)
    grayscale_image[1::2, 1::2] = rggb_image[:, :, 3]  # B 채널

    return grayscale_image

def visualize_RAW_image(raw_image, output_file_name):
    grayscale_image = rggb_to_grayscale(raw_image)
    output_image = Image.fromarray(grayscale_image)
    output_image.save(output_file_name)


# referenced by https://github.com/princeton-vl/RAFT/issues/64
def backward_warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    
    print('backward warping',grid.shape)
    
    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid)
    mask = torch.ones(x.size()).to('cpu')
    mask = F.grid_sample(mask, vgrid)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    return output


def forward_warp(image, flow):
    """
    Forward warps an image using the provided optical flow.
    
    Args:
        image (torch.Tensor): Input images of shape (B, C, H, W).
        flow (torch.Tensor): Optical flow of shape (B, 2, H, W).
        
    Returns:
        torch.Tensor: Forward warped images of shape (B, C, H, W).
    """
    B, C, H, W = image.shape
    device = image.device

    # Initialize the output image and weight map
    warped_image = torch.zeros_like(image)
    weight_map = torch.zeros(B, 1, H, W, device=device)

    # Create a meshgrid of pixel coordinates
    y_grid, x_grid = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device))
    y_grid = y_grid.unsqueeze(0).expand(B, -1, -1)  # Shape: (B, H, W)
    x_grid = x_grid.unsqueeze(0).expand(B, -1, -1)  # Shape: (B, H, W)

    # Compute target coordinates
    x_flow = x_grid + flow[:, 0, :, :]
    y_flow = y_grid + flow[:, 1, :, :]

    # Round coordinates to nearest integer
    x_flow = x_flow.round().long()
    y_flow = y_flow.round().long()

    # Clip coordinates to image boundaries
    x_flow = x_flow.clamp(0, W - 1)
    y_flow = y_flow.clamp(0, H - 1)

    # Flatten tensors for indexing
    batch_indices = torch.arange(B, device=device).view(B, 1).expand(-1, H * W).flatten()
    src_indices = (y_grid * W + x_grid).view(B, -1).long()
    tgt_indices = (y_flow * W + x_flow).view(B, -1).long()

    # Flatten images
    image_flat = image.view(B, C, -1)

    # Accumulate pixel values at target locations
    for b in range(B):
        warped_image_flat = warped_image[b].view(C, -1)
        weight_map_flat = weight_map[b].view(1, -1)

        # Use scatter_add to handle multiple pixels mapping to the same location
        warped_image_flat.scatter_add_(1, tgt_indices[b].unsqueeze(0).expand(C, -1), image_flat[b])
        weight_map_flat.scatter_add_(1, tgt_indices[b].unsqueeze(0), torch.ones(1, H * W, device=device))

    # Normalize the accumulated values by the weights to handle overlaps
    weight_map = weight_map.clamp(min=1.0)  # Prevent division by zero
    warped_image = warped_image / weight_map

    return warped_image





def process_images_and_flow(input_images, input_flows, warp_type):
    """
    주어진 이미지를 Optical flow를 사용해 backward warping하여 정렬합니다.
    
    매개변수:
    - input_images: Concatenated 이미지들 (3x3xHxW)
    - input_flows: Concatenated optical flows (2x2xHxW)
    
    반환값:
    - Aligned images (aligned_image_t_minus_1, aligned_image_t_plus_1)
    """
    # 이미지 분리 (3x3xHxW -> 각각 3xHxW)
    image_t_minus_1 = input_images[0][0]  # t-1 이미지 (3xHxW)
    image_t = input_images[0][1]          # t 이미지 (3xHxW)
    image_t_plus_1 = input_images[0][2]   # t+1 이미지 (3xHxW)
    
    # Optical flow 분리 (2x2xHxW -> 각각 2xHxW)
    flow_t_to_t_minus_1 = input_flows[0][0]  # t에서 t-1로 가는 optical flow (2xHxW)
    flow_t_to_t_plus_1 = input_flows[0][1]   # t에서 t+1로 가는 optical flow (2xHxW)

    image_t_minus_1 = image_t_minus_1.unsqueeze(0)
    flow_t_to_t_minus_1 = flow_t_to_t_minus_1.unsqueeze(0)
    image_t_plus_1 = image_t_plus_1.unsqueeze(0)
    flow_t_to_t_plus_1 = flow_t_to_t_plus_1.unsqueeze(0)
    
    print(image_t_minus_1.shape, flow_t_to_t_minus_1.shape, image_t_plus_1.shape, flow_t_to_t_plus_1.shape)

    if warp_type == 'backward':
        # 이미지 t-1을 t에 맞게 정렬
        aligned_image_t_minus_1 = backward_warp(image_t_minus_1, flow_t_to_t_minus_1)
        # aligned_image_t_minus_1 = warp_image(image_t_minus_1, flow_t_to_t_minus_1)
        
        # 이미지 t+1을 t에 맞게 정렬
        aligned_image_t_plus_1 = backward_warp(image_t_plus_1, flow_t_to_t_plus_1)
    elif warp_type == 'forward':
        aligned_image_t_minus_1 = forward_warp(image_t_minus_1, flow_t_to_t_minus_1)
        aligned_image_t_plus_1 = forward_warp(image_t_plus_1, flow_t_to_t_plus_1)
    # aligned_image_t_plus_1 = warp_image(image_t_plus_1, flow_t_to_t_plus_1)
    
    return aligned_image_t_minus_1, image_t, aligned_image_t_plus_1

def save_tensor_as_png(tensor, filename):
    """
    PyTorch tensor를 PNG 파일로 저장합니다.
    
    매개변수:
    - tensor: 저장할 PyTorch tensor (C, H, W) 또는 (H, W)
    - filename: 저장할 PNG 파일 이름
    """
    # 텐서 shape 변경 (1, C, H, W) -> (C, H, W)
    tensor = tensor.squeeze(0) 
    
    # 텐서를 numpy로 변환 (C, H, W) -> (H, W, C)
    if tensor.ndim == 3:
        tensor = tensor.permute(1, 2, 0).cpu().numpy()
    elif tensor.ndim == 2:
        tensor = tensor.cpu().numpy()
    
    # 값의 범위를 [0, 255]로 조정
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min()) * 255
    tensor = tensor.astype(np.uint8)
    
    # PIL 이미지를 생성하여 저장
    img = Image.fromarray(tensor)
    img.save(filename)



def save_tensor_as_png(tensor, filename):
    """
    PyTorch tensor를 PNG 파일로 저장합니다.
    
    매개변수:
    - tensor: 저장할 PyTorch tensor (C, H, W) 또는 (H, W)
    - filename: 저장할 PNG 파일 이름
    """
    # 텐서 shape 변경 (1, C, H, W) -> (C, H, W)
    tensor = tensor.squeeze(0) 
    
    # 텐서를 numpy로 변환 (C, H, W) -> (H, W, C)
    if tensor.ndim == 3:
        tensor = tensor.permute(1, 2, 0).cpu().numpy()
    elif tensor.ndim == 2:
        tensor = tensor.cpu().numpy()
    
    # 값의 범위를 [0, 255]로 조정
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min()) * 255
    tensor = tensor.astype(np.uint8)
    
    # PIL 이미지를 생성하여 저장
    img = Image.fromarray(tensor)
    img.save(filename)

def show_images_from_files(filenames):
    """
    PNG 파일을 읽고 Jupyter Notebook에서 한 번에 시각화합니다.
    
    매개변수:
    - filenames: 시각화할 PNG 파일 이름 리스트
    """    
    plt.figure(figsize=(20, 6))

    for i, filename in enumerate(filenames):
        print(filename)
        img = Image.open(filename)
        plt.subplot(1, 3, i+1)
        plt.imshow(img)
        if i==0: plt.title('t-1 frame')
        elif i==1: plt.title('t frame')
        elif i==2: plt.title('t+1 frame')
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    
def resize_optical_flow(flow):
    """
    주어진 HxWx2 크기의 optical flow를 H/2xW/2x2 크기로 축소하고,
    각 flow 원소를 0.5배로 조정하는 함수.

    Parameters:
    - flow: HxWx2 크기의 optical flow (u, v)

    Returns:
    - resized_flow: H/2xW/2x2 크기의 optical flow
    """
    # Optical flow의 HxWx2 크기를 H/2xW/2x2로 축소
    height, width = flow.shape[:2]
    
    # 이미지를 절반 크기로 축소
    resized_flow = cv2.resize(flow, (width // 2, height // 2), interpolation=cv2.INTER_LINEAR)

    # 각 flow 요소를 0.5배로 축소 (픽셀 크기 변화에 따른 보정)
    resized_flow *= 0.5

    return resized_flow

import flow_vis

def flow_to_color_map(flow, rgb2bgr=True, max_flow=None):    
    # flow_uv = torch.from_numpy(flow).float()
    flow_rgb = flow_vis.flow_to_color(flow, convert_to_bgr=rgb2bgr)
    return flow_rgb

def visualize_and_save(tensor1, tensor2, flow, save_path):
    # tensor1과 tensor2는 [C, H, W] 형태, flow는 [2, H, W] 형태
    C, H, W = tensor1.shape
    
    # 텐서와 optical flow를 numpy 형태로 변환
    img1 = tensor1.permute(1, 2, 0).cpu().numpy()
    img2 = tensor2.permute(1, 2, 0).cpu().numpy()
    
    flow_np = flow.permute(1, 2, 0).cpu().numpy()
    
    # 이미지 범위를 [0, 255]로 변환
    img1 = (img1 * 255).astype(np.uint8)
    img2 = (img2 * 255).astype(np.uint8)
    
    # optical flow 시각화
    flow_color = flow_to_color_map(flow_np)
    images = [img1, img2, flow_color]
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        ax[i].imshow(images[i])
        ax[i].axis('off')  # 축 제거
    
    
    # 서브플롯 간 여백 제거
    plt.subplots_adjust(wspace=0, hspace=0)
    
    # 이미지 저장
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

