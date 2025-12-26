import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import cv2
from typing import Union, List, Dict

# import flow_vis_torch
import flow_vis

def visualize_optical_flow(image, flow_u, flow_v, step=16):
    h, w = image.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].astype(np.int64)
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.quiver(x, y, flow_u[y, x], flow_v[y, x], color='r', angles='xy', scale_units='xy', scale=1, width=0.0025)
    ax.set_title('Optical Flow Visualization')
    ax.axis('off')
    return fig

def draw_and_save_images(image_tensor, flow_tensor, save_path):
    """
    이미지 2개와 Optical Flow를 시각화하여 하나의 이미지로 저장하는 함수
    
    :param image_tensor: (B, 3, C, H, W) 형태의 이미지 텐서
    :param flow_tensor: (B, 2, 2, H, W) 형태의 optical flow 텐서
    :param save_path: 저장할 이미지 경로
    """
    # 첫 번째 batch만 사용
    img0 = image_tensor[0, 0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    img1 = image_tensor[0, 1].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    flow_u = flow_tensor[0, 0, 0].cpu().numpy()
    flow_v = flow_tensor[0, 0, 1].cpu().numpy()

    # 이미지 시각화
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 첫 번째 이미지 출력
    axes[0].imshow(img0)
    axes[0].set_title('Image 1')
    axes[0].axis('off')

    # 두 번째 이미지 출력
    axes[1].imshow(img1)
    axes[1].set_title('Image 2')
    axes[1].axis('off')

    # Optical Flow 시각화 (첫 번째 이미지를 배경으로 사용)
    axes[2].imshow(img0)
    y, x = np.mgrid[16//2:img0.shape[0]:16, 16//2:img0.shape[1]:16].astype(np.int64)
    axes[2].quiver(x, y, flow_u[y, x], flow_v[y, x], color='r', angles='xy', scale_units='xy', scale=1, width=0.0025)
    axes[2].set_title('Optical Flow Visualization')
    axes[2].axis('off')

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def process_images_and_flow(image_tensor, flow_tensor, save_path):
    """
    외부에서 호출되는 함수 예시. 이미지 텐서와 optical flow 텐서를 받아 시각화를 저장.
    
    :param image_tensor: (B, 3, C, H, W) 형태의 이미지 텐서
    :param flow_tensor: (B, 2, 2, H, W) 형태의 optical flow 텐서
    :param save_path: 시각화된 이미지를 저장할 경로
    """
    draw_and_save_images(image_tensor, flow_tensor, save_path)

# Example usage (외부에서 호출 시)
def example_usage():
    # Example input tensor
    batch_size = 1
    channels = 3
    height = 360
    width = 640
    T = 3
    image_tensor = torch.rand((batch_size, T, channels, height, width)) * 255
    image_tensor = image_tensor.type(torch.uint8).cuda()

    # Example flow tensor
    flow_tensor = torch.rand((batch_size, 2, 2, height, width)).cuda()

    # Save path
    save_path = '/path/to/save/optical_flow_visualization.png'
    
    # Call the processing function
    process_images_and_flow(image_tensor, flow_tensor, save_path)

def save_images_as_grid(image_tensor, save_path, img_name):
    """
    (B, T, C, H, W) 형태의 텐서에서 첫 번째 배치의 T 개 이미지를 가로로 이어 붙여 저장하는 함수.
    
    :param image_tensor: (B, T, C, H, W) 형태의 이미지 텐서
    :param save_path: 저장할 이미지 파일 경로
    """
    # 첫 번째 배치의 이미지들만 사용
    batch_images = image_tensor[0]  # Shape: (T, C, H, W)
    
    
    # 서브플롯 생성 (1행 T열)
    T = batch_images.size(0)
    fig, ax = plt.subplots(1, T, figsize=(T * 3, 3))  # T개의 가로 서브플롯 생성
    
    # 각 서브플롯에 이미지 추가
    for i in range(T):
        # 이미지를 (H, W, C) 형태로 변환하고 numpy 배열로 변환
        img = (batch_images[i] * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        ax[i].imshow(img)
        ax[i].axis('off')  # 축 제거
        
        
        cv2.imwrite(f'{save_path}/{img_name}_{i}.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    # 서브플롯 간 여백 제거
    plt.subplots_adjust(wspace=0, hspace=0)
    
    # 이미지 저장
    plt.savefig(save_path+'/grid_image.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    
def flow_to_color_map(flow, max_flow=None):    
    # flow_uv = torch.from_numpy(flow).float()
    flow_rgb = flow_vis.flow_to_color(flow, convert_to_bgr=False)
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



def update_out_vis(out_vis, key, rgb, raw, flow):
    """
    특정 key에 대해 rgb, raw, flow 텐서를 저장하는 함수.

    Parameters:
    - out_vis (dict): 저장할 dictionary
    - key (str): 저장할 key (예: 'MRRB_encoder_3')
    - rgb (torch.Tensor): [B, C, T, H, W] 크기의 텐서
    - raw (torch.Tensor): [B, C, T, H, W] 크기의 텐서
    - flow (torch.Tensor): [B, C, T, H, W] 크기의 텐서 (T=2)
    """

    # Batch index 0 기준으로 C, T, H, W 로 변환
    rgb = rgb[0]  # [C, T, H, W]
    raw = raw[0]  # [C, T, H, W]
    flow = flow[0]  # [C, T, H, W]

    # C, T, H, W -> T 개의 C, H, W 로 분할
    rgb_tm, rgb_t, rgb_tp = rgb.split(1, dim=1)  # [C, 1, H, W] 각각 3개
    raw_tm, raw_t, raw_tp = raw.split(1, dim=1)  # [C, 1, H, W] 각각 3개
    flow_t_to_tm, flow_t_to_t, flow_t_to_tp = flow.split(1, dim=1)  # [C, 1, H, W] 각각 2개

    # C, 1, H, W -> 1, H, W 로 변환 (C 축 평균)
    def process_tensor(tensor):
        return tensor.mean(dim=0, keepdim=False).detach().cpu()  # [1, H, W]

    out_vis[key]['rgb_tm'] = process_tensor(rgb_tm)
    out_vis[key]['rgb_t'] = process_tensor(rgb_t)
    out_vis[key]['rgb_tp'] = process_tensor(rgb_tp)

    out_vis[key]['raw_tm'] = process_tensor(raw_tm)
    out_vis[key]['raw_t'] = process_tensor(raw_t)
    out_vis[key]['raw_tp'] = process_tensor(raw_tp)

    out_vis[key]['flow_t_to_tm'] = process_tensor(flow_t_to_tm)
    out_vis[key]['flow_t_to_t'] = process_tensor(flow_t_to_t)
    out_vis[key]['flow_t_to_tp'] = process_tensor(flow_t_to_tp)



def reshape_tensor(tensor: torch.Tensor, data_type: str, num_flow: int) -> torch.Tensor:
    """
    Reshape tensor based on its input shape.
    - If (B, C, 3, H, W): concatenate frames horizontally to (B, C, H, 3W), then reshape using num_flow.
    - If (B, C, H, W): apply num_flow directly to reshape to (B, C//num_flow, H*num_flow, W).
    """
    # Detach tensor and move to CPU
    tensor = tensor.detach().cpu()
    
    if tensor.dim() == 5:  # Case when input is (B, C, 3, H, W)
        B, C, F, H, W = tensor.shape
        assert F == 3, "The input tensor must have 3 frames in the third dimension."
        
        # Reshape to (B, C, H, 3W) by concatenating frames horizontally
        tensor = tensor.permute(0, 1, 3, 2, 4).reshape(B, C, H, F * W)
        W = F * W
        
        # Reshape based on num_flow
        assert C % num_flow == 0, "C must be divisible by num_flow."
        C_new = C // num_flow
        tensor = tensor.reshape(B, num_flow, C_new, H, W)
        tensor = tensor.permute(0, 2, 1, 3, 4)
        tensor = tensor.reshape(B, C_new, num_flow * H, W)
        
    elif tensor.dim() == 4:  # Case when input is (B, C, H, W)
        B, C, H, W = tensor.shape
        
    else:
        raise ValueError("Input tensor must have 4 or 5 dimensions.")
    
    # Reshape based on num_flow
    assert C % num_flow == 0, "C must be divisible by num_flow."
    C_new = C // num_flow
    tensor = tensor.reshape(B, num_flow, C_new, H, W)
    tensor = tensor.permute(0, 2, 1, 3, 4)
    tensor = tensor.reshape(B, C_new, num_flow * H, W)
    
    return tensor

"""
* 항상 기본 입력은 b c t h w 이다 / b c h w 도 가능
* data_type 의 경우 rgb, raw, flow 중 하나를 사용
  * rgb: 3 channel image || c channel feature
  * raw: 4 channel image || c channel feature
  * 위의 rgb, raw 의 경우 각각 3, 4 채널을 받을수도 있고 만약 3, 4 채널이 아닌 경우에는 feature map 으로 간주하여 1채널로 평균을 내고 해당 feauture map 을 저장한다.
  * 즉 rgb, raw 각각에서 feature 와 image 를 구분하여 처리한다.
  * 이 과정은 visualize_pth_to_png_type2.py 에서 실행된다.
  * 아래 코드는 단순히 feature 와 data_type 을 저장만 하는 코드이다.
"""

def store_vis_data(
    storage_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
    keys: Union[str, List[str]],
    data: Union[torch.Tensor, List[torch.Tensor]],
    data_type: str,
    num_flow: int = 1
):
    """
    Store visualization data into the given dictionary with reshaped tensors.
    
    Args:
        storage_dict (dict): The dictionary to store tensors.
        keys (str or List[str]): The key or list of keys under which the data is stored.
        data (torch.Tensor or List[torch.Tensor]): The data tensor(s) to be stored.
        data_type (str): Type of data (rgb, raw, flow).
        num_flow (int): Number of flow groups for reshaping.
    """
    assert data_type in {"rgb", "raw", "flow"}, "Invalid data type."
    
    if isinstance(keys, list) and isinstance(data, list):
        assert len(keys) == len(data), "Keys and data lists must have the same length."
        storage_dict.update({key: {'data':reshape_tensor(d, data_type, num_flow), 'num_flow':num_flow, 'data_type':data_type} for key, d in zip(keys, data)})
    elif isinstance(keys, str) and isinstance(data, torch.Tensor):
        storage_dict[keys] = {'data':reshape_tensor(data, data_type, num_flow), 'num_flow':num_flow, 'data_type':data_type}
    else:
        raise ValueError("Keys and data must both be either lists or single values.")