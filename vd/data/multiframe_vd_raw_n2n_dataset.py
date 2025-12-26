import torch
from torch.utils import data as data

from vd.data.data_util import *
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

import numpy as np
    
@DATASET_REGISTRY.register()
class MultiFrameVDPairedN2NRawDataset(data.Dataset):
    def __init__(self, opt):
        super(MultiFrameVDPairedN2NRawDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']

        if self.opt['phase'] == 'train':
            self.paths = video_rawrgb_3_to_3_frames_from_folders_train([self.lq_folder, self.gt_folder])
        else:
            self.paths = video_rawrgb_3_to_3_frames_from_folders_val([self.lq_folder, self.gt_folder])

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        key = self.paths[index]['key']

        # Load gt image and alignratio.
        gt_0_path = self.paths[index]['gt_rgb_0_path']
        img_gt_rgb_0 = read_img(gt_0_path)
        
        gt_1_path = self.paths[index]['gt_rgb_1_path']
        img_gt_rgb_1 = read_img(gt_1_path)
        
        gt_2_path = self.paths[index]['gt_rgb_2_path']
        img_gt_rgb_2 = read_img(gt_2_path)


        # Load lq images.
        lq_0_path = self.paths[index]['lq_raw_0_path']
        img_lq_raw_0 = read_npz(lq_0_path)

        lq_1_path = self.paths[index]['lq_raw_1_path']
        img_lq_raw_1 = read_npz(lq_1_path)

        lq_2_path = self.paths[index]['lq_raw_2_path']
        img_lq_raw_2 = read_npz(lq_2_path)
        
        img_results = [img_gt_rgb_0, img_gt_rgb_1, img_gt_rgb_2, img_lq_raw_0, img_lq_raw_1, img_lq_raw_2]

        if self.opt['phase'] == 'train':
            rgb_size_imgs, raw_size_imgs = paired_random_crop(img_results[:3], img_results[3:6], self.opt['crop_size'], 2)
            imgs = [rgb_size_imgs[0], rgb_size_imgs[1], rgb_size_imgs[2], raw_size_imgs[0], raw_size_imgs[1], raw_size_imgs[2]]
            imgs = augment(imgs, self.opt['use_hflip'], self.opt['use_rot'])
            
            # BGR to RGB, HWC to CHW, numpy to tensor
            img_gt_rgb_0, img_gt_rgb_1, img_gt_rgb_2 = img2tensor(imgs[0:3], bgr2rgb=True, float32=True)
            img_lq_raw_0, img_lq_raw_1, img_lq_raw_2 = img2tensor(imgs[3:6], bgr2rgb=True, float32=True)
            
        else:
            img_gt_rgb_0, img_gt_rgb_1, img_gt_rgb_2 = img2tensor(img_results[0:3], bgr2rgb=True, float32=True)
            img_lq_raw_0, img_lq_raw_1, img_lq_raw_2 = img2tensor(img_results[3:6], bgr2rgb=True, float32=True)
        

        # stack the tensor to match the shape as [T, C, H, W]
        img_raw_lqs = torch.stack([img_lq_raw_0, img_lq_raw_1, img_lq_raw_2], dim=0) # [T=3, C=4, H, W]
        img_raw_gts = torch.stack([img_gt_rgb_0, img_gt_rgb_1, img_gt_rgb_2], dim=0) # [T=3, C=4, H, W]
        return {'lq': img_raw_lqs, 'gt': img_raw_gts, 'key': key}

    def __len__(self):
        return len(self.paths)