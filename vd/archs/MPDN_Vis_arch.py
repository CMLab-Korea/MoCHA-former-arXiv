"""
Moire Pattern Disentanglement Network (MPDN)
MDPN_ver1: fusing module uses cat -> 1x1 conv
MDPN_ver2: fusing module uses add -> 1x1 conv
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from vd.archs.base_utils_arch import MoireGenerator, ConvNeXtBlocks
from vd.archs.cross_attn_utils_arch import FCAM_ver1, FCAM_ver2
from vd.utils.test_visualize import store_vis_data
# from vd.archs.PSRT_arch import RSTB, PatchEmbed, compute_mask
# from vd.archs.vid_module_utils_arch import FCAM3D, ConvNeXt3DBlocks
        
# using sigmoid (instead of softmax) for attention map
# similar to MPDM_ver1
class MPDM_ver3(nn.Module):
    def __init__(self, embed_dim=96, num_frames=3, convnext_block_m=3, convnext_block_c=3, stage=1):
        super(MPDM_ver3, self).__init__()
        self.num_frames = num_frames
        self.stage = stage
        self.conv_isp = nn.Sequential(
                            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim // 2, kernel_size=1, stride=1, padding=0), # 1x1 conv
                            nn.Conv2d(in_channels=embed_dim // 2, out_channels=4 * 3, kernel_size=3, stride=1, padding=1), # 3x3 conv
                            nn.PixelShuffle(upscale_factor=2)) # pixel shuffle
        
        
        # Coarse Demoireing Module
        self.CDM = ConvNeXtBlocks(embed_dim, convnext_block_m)
        self.conv_last_m = nn.Sequential(
                                nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim // 2, kernel_size=1, stride=1, padding=0), # 1x1 conv
                                nn.GELU(),
                                nn.Conv2d(in_channels=embed_dim // 2, out_channels=embed_dim // 4, kernel_size=3, stride=1, padding=1), # 3x3 conv
                                nn.GELU(),
                                nn.Conv2d(in_channels=embed_dim // 4, out_channels=4, kernel_size=3, stride=1, padding=1), # 3x3 conv
                            )
        # Moire Pattern Predicting Module
        self.MPM = ConvNeXtBlocks(embed_dim, convnext_block_c)
        self.conv_last_c = nn.Sequential(
                                nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim // 2, kernel_size=1, stride=1, padding=0), # 1x1 conv
                                nn.GELU(),
                                nn.Conv2d(in_channels=embed_dim // 2, out_channels=embed_dim // 4, kernel_size=3, stride=1, padding=1), # 3x3 conv
                                nn.GELU(),
                                nn.Conv2d(in_channels=embed_dim // 4, out_channels=4, kernel_size=3, stride=1, padding=1), # 3x3 conv
                            )
        
        self.conv_last_c_rgb = nn.Sequential(
                                nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim // 2, kernel_size=1, stride=1, padding=0), # 1x1 conv
                                nn.GELU(),
                                nn.Conv2d(in_channels=embed_dim // 2, out_channels=embed_dim, kernel_size=3, stride=1, padding=1), # 3x3 conv
                            )
        
        self.fcam = FCAM_ver1(embed_dim, num_heads=4, ffn_expansion_factor=2.0, bias=False, LayerNorm_type='WithBias')
    
    def forward(self, x, vis_state=False):
        nt, c, h, w = x.size()
        t = self.num_frames
        n = nt // t
        if vis_state: 
            vis_dict = {
                '02_coarse_clean_feat': None,
                '02_moire_feat': None,
                '02_masked_feat': None,
                '02_attn_feat': None,
            }

        # moire pattern predictor + coarse clean frame predictor
        moire_pat = x + self.MPM(x) # bt c h w
        clean_inter_frm = x + self.CDM(x) # bt c h w
        
        ## vis part #######################################################################################
        if vis_state:
            store_vis_data(vis_dict, '02_moire_feat', rearrange(moire_pat, '(b t) c h w -> b c t h w', b=n, t=t).contiguous(), 'raw', num_flow=1)
            store_vis_data(vis_dict, '02_coarse_clean_feat', rearrange(clean_inter_frm, '(b t) c h w -> b c t h w', b=n, t=t).contiguous(), 'raw', num_flow=1)
        ###################################################################################################        
        
        moire_pat_out = self.conv_last_m(moire_pat)
        clean_inter_frm_out = self.conv_last_c(clean_inter_frm)
        
        # convert raw feature to sRGB frame
        clean_inter_frm_out_rgb = self.conv_last_c_rgb(clean_inter_frm)
        clean_inter_frm_out_rgb = self.conv_isp(clean_inter_frm_out_rgb)
                   
        # apply cross attention with moire patterns and coarse clean frames / 
        # Q: clean_inter_frm / conv blocks, KV: moire_pat / conv blocks
        fcam_feat = self.fcam(clean_inter_frm, moire_pat) # bt c h w
        # apply 1x1 conv to reduce channel dim -> for generating per-frame attention map
        
        spa_attn = F.sigmoid(fcam_feat) # bt c h w
        
        moire_pat = rearrange(moire_pat, '(b t) c h w -> b c t h w', b=n, t=t).contiguous() # b c t h w
        clean_inter_frm = rearrange(clean_inter_frm, '(b t) c h w -> b c t h w', b=n, t=t).contiguous() # b c t h w
        spa_attn = rearrange(spa_attn, '(b t) c h w -> b c t h w', b=n, t=t).contiguous() # b c t h w

        # apply attention map to coarse clean frame (+ skip connection)
        clean_inter_frm = clean_inter_frm * spa_attn + clean_inter_frm     
        
        ## vis part #######################################################################################
        if vis_state:
            store_vis_data(vis_dict, '02_attn_feat', spa_attn, 'raw', num_flow=1)
            store_vis_data(vis_dict, '02_masked_feat', clean_inter_frm, 'raw', num_flow=1)
        ###################################################################################################
            
        # coarse_feat = self.conv_1x1(clean_inter_frm)
        coarse_feat = rearrange(clean_inter_frm, 'b c t h w -> (b t) c h w', b=n, t=t).contiguous() # b t c h w
        
        """
        coarse_feat: 
            * coarse demoired feature / output of MPDN
            * (bt c h w) / c == embed_dim(formally 96)
        moire_pat_out: 
            * disentangled moire pattern
            * (bt c h w) / c == 4
        clean_inter_frm_out: 
            * disentangled clean frame (Raw domain)
            * (bt c h w) / c == 4
        clean_inter_frm_out_rgb: 
            * disentangled clean frame / translated into sRGB space using ISP (sRGB domain)
            * (bt c h w) / c == 3
        """
        if vis_state:
            return coarse_feat, moire_pat_out, clean_inter_frm_out, clean_inter_frm_out_rgb, vis_dict
        elif vis_state == False:
            return coarse_feat, moire_pat_out, clean_inter_frm_out, clean_inter_frm_out_rgb   
             
