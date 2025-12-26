from thop import profile


import torch
import torch.nn as nn
from ptflops import get_model_complexity_info
import time
from deepspeed.profiling.flops_profiler import get_model_profile

from vd.archs.PSRT_arch import PSRT
from vd.archs.PSRT_CC_ver2_arch import PSRT_CC_ver2
from vd.archs.PSRT_CCMS_arch import PSRT_CCMS
from vd.archs.PSRT_CCMS2_arch import PSRT_CCMS2
from vd.archs.PSRT_CCMS3_arch import PSRT_CCMS3
from vd.archs.PSRT_CCMS4_arch import PSRT_CCMS4
from vd.archs.PSRT_CCMS5_arch import PSRT_CCMS5
from vd.archs.PSRT_CCMS6_Vis_arch import PSRT_CCMS6_Vis
from vd.archs.PSRT_CCMS7_Vis_arch import PSRT_CCMS7_Vis
from vd.archs.PSRT_CCMS9_Vis_arch import PSRT_CCMS9_Vis
from vd.archs.PSRT_CCMS10_Vis_arch import PSRT_CCMS10_Vis
from vd.archs.PSRT_CCMS11_Vis_arch import PSRT_CCMS11_Vis
from vd.archs.PSRT_CCMS12_Vis_arch import PSRT_CCMS12_Vis
from vd.archs.PSRT_CCMS13_Vis_arch import PSRT_CCMS13_Vis
from vd.archs.PSRT_CCMS14_Vis_arch import PSRT_CCMS14_Vis
from vd.archs.PSRT_CCMS16_Vis_arch import PSRT_CCMS16_Vis
from vd.archs.PSRT_CCMS17_Vis_arch import PSRT_CCMS17_Vis
from vd.archs.PSRT_CCMS18_Vis_arch import PSRT_CCMS18_Vis
from vd.archs.PSRT_CCMS7_3_Vis_arch import PSRT_CCMS7_3_Vis
from vd.archs.PSRT_CCMS7_4_Vis_arch import PSRT_CCMS7_4_Vis
from vd.archs.PSRT2_arch import PSRT2
from vd.archs.PSRT3_arch import PSRT3
from vd.archs.PSRT4_arch import PSRT4
from vd.archs.PSRT7_arch import PSRT7
from vd.archs.PSRT8_arch import PSRT8
from vd.archs.PSRT9_arch import PSRT9
from vd.archs.PSRT_CCMS6_4_Vis_arch import PSRT_CCMS6_4_Vis
from vd.archs.PSRT_CCMS6_6_Vis_arch import PSRT_CCMS6_6_Vis
from vd.archs.PSRT_CCMS6_7_Vis_arch import PSRT_CCMS6_7_Vis
from vd.archs.PSRT_CCMS6_8_Vis_arch import PSRT_CCMS6_8_Vis
from vd.archs.PSRT_CCMS6_9_Vis_arch import PSRT_CCMS6_9_Vis
from vd.archs.MoCHAformer_arch import MoCHAformer
from vd.archs.MoCHAformer4_arch import MoCHAformer4

from vd.archs.MoCHAformer4_ab4_2_arch import MoCHAformer4_ab4_2
from vd.archs.MoCHAformer4_ab4_3_arch import MoCHAformer4_ab4_3
from vd.archs.MoCHAformer4_ab4_4_arch import MoCHAformer4_ab4_4
from vd.archs.MoCHAformer4_ab4_5_arch import MoCHAformer4_ab4_5


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# model = PSRT(img_size=128, patch_size=1, in_chans=4, embed_dim=24, 
#                     depths=[5,5,5,5], num_heads=[4,4,4,4], window_size=[2,7,7], 
#                     mlp_ratio=2, qkv_bias=True, qk_scale=False, 
#                     drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
#                     ape=False, patch_norm=True, use_checkpoint=False, 
#                     upscale=1, img_range=1., upsampler='pixelshuffle', resi_connection='1conv', num_frames=3).to(device)

# model = PSRT_Freq(img_size=128, patch_size=1, in_chans=4, embed_dim=96, 
#                     depths=[5,5,5,5], num_heads=[4,4,4,4], window_size=[2,7,7], 
#                     mlp_ratio=2, qkv_bias=True, qk_scale=False, 
#                     drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
#                     ape=False, patch_norm=True, use_checkpoint=False, 
#                     upscale=1, img_range=1., upsampler='pixelshuffle', resi_connection='1conv', num_frames=3).to(device)

# model = PSRT_MulRes(img_size=128, patch_size=1, in_chans=4, embed_dim=48, 
#                     depths=[5,5,5,5], num_heads=[4,4,4,4], window_size=[2,7,7], 
#                     mlp_ratio=2, qkv_bias=True, qk_scale=False, 
#                     drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
#                     ape=False, patch_norm=True, use_checkpoint=False, 
#                     upscale=1, img_range=1., upsampler='pixelshuffle', resi_connection='1conv', num_frames=3).to(device)

# model = PSRT_Freq_v2(img_size=256, patch_size=1, in_chans=4, embed_dim=96, 
#                     depths=[5,5,5,5], num_heads=[4,4,4,4], window_size=[2,7,7], 
#                     mlp_ratio=2, qkv_bias=True, qk_scale=False, 
#                     drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
#                     ape=False, patch_norm=True, use_checkpoint=False, 
#                     upscale=1, img_range=1., upsampler='pixelshuffle', resi_connection='1conv', num_frames=3).to(device)

# model = PSRT_Freq_v7(img_size=256, patch_size=1, in_chans=4, embed_dim=96, 
#                     depths=[3,3,3,3], num_heads=[4,4,4,4], window_size=[2,7,7], 
#                     mlp_ratio=2, qkv_bias=True, qk_scale=False, 
#                     drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
#                     ape=False, patch_norm=True, use_checkpoint=False, 
#                     upscale=1, img_range=1., upsampler='pixelshuffle', resi_connection='1conv', num_frames=3).to(device)

# model = PSRT_Freq_v8(img_size=256, patch_size=1, in_chans=4, embed_dim=96, 
#                     depths=[3,3,3,3], num_heads=[4,4,4,4], window_size=[2,7,7], 
#                     mlp_ratio=2, qkv_bias=True, qk_scale=False, 
#                     drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
#                     ape=False, patch_norm=True, use_checkpoint=False, 
#                     upscale=1, img_range=1., upsampler='pixelshuffle', resi_connection='1conv', num_frames=3).to(device)

# model = PSRT(img_size=256, patch_size=1, in_chans=4, embed_dim=96, 
#             depths=[5,5,5], num_heads=[4,4,4], window_size=[2,7,7], 
#             mlp_ratio=2, qkv_bias=True, qk_scale=False, 
#             drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
#             ape=False, patch_norm=True, use_checkpoint=False, 
#             upscale=1, img_range=1., upsampler='pixelshuffle', resi_connection='1conv', num_frames=3).to(device)

# model = PSRT_CL_demoire(convnext_block=3, img_size=256, patch_size=1, in_chans=4, embed_dim=96, 
#             depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#             mlp_ratio=2, qkv_bias=True, qk_scale=False, 
#             drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
#             ape=False, patch_norm=True, use_checkpoint=False, 
#             upscale=1, img_range=1., upsampler='pixelshuffle', resi_connection='1conv', num_frames=3).to(device)

# model = PSRT_CL_moire(convnext_block=3, in_chans_rgb=3, in_chans_raw=4, embed_dim=96).to(device)

# model = PSRT_CC(convnext_block_c=3, convnext_block_m=3, img_size=128, patch_size=1, in_chans=4, embed_dim=96, 
#             depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#             mlp_ratio=2, qkv_bias=True, qk_scale=False, 
#             drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
#             ape=False, patch_norm=True, use_checkpoint=False, 
#             upscale=1, img_range=1., upsampler='pixelshuffle', resi_connection='1conv', num_frames=3).to(device)

# model = PSRT_CL_demoire(convnext_block=3, img_size=128, patch_size=1, in_chans=4, embed_dim=96, 
#             depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#             mlp_ratio=2, qkv_bias=True, qk_scale=False, 
#             drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
#             ape=False, patch_norm=True, use_checkpoint=False, 
#             upscale=1, img_range=1., upsampler='pixelshuffle', resi_connection='1conv', num_frames=3).to(device)

# model = PSRT_CL_moire(convnext_block=3, in_chans_rgb=3, in_chans_raw=4, embed_dim=96).to(device)

# model = PSRT_CC_ver2(convnext_block_c=3,convnext_block_m=3,img_size=128,patch_size=1,
#                      in_chans=4,embed_dim=96,
#                      depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#                      mlp_ratio=2.,qkv_bias=True,qk_scale=None,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.1,
#                      norm_layer=nn.LayerNorm,ape=False,patch_norm=True,use_checkpoint=False,upscale=1,img_range=1.,
#                      upsampler='pixelshuffle',resi_connection='1conv',num_frames=3).to(device)

# model = PSRT_CCMS(stage=1, 
#                     convnext_block_c=3,convnext_block_m=3,img_size=128,patch_size=1,
#                     in_chans=4,embed_dim=96,
#                     depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#                     mlp_ratio=2.,qkv_bias=True,qk_scale=None,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.1,
#                     norm_layer=nn.LayerNorm,ape=False,patch_norm=True,use_checkpoint=False,upscale=1,img_range=1.,
#                     upsampler='pixelshuffle',resi_connection='1conv',num_frames=3).to(device)

# model = PSRT_CCMS(stage=2, 
#                     convnext_block_c=3,convnext_block_m=3,img_size=128,patch_size=1,
#                     in_chans=4,embed_dim=96,
#                     depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#                     mlp_ratio=2.,qkv_bias=True,qk_scale=None,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.1,
#                     norm_layer=nn.LayerNorm,ape=False,patch_norm=True,use_checkpoint=False,upscale=1,img_range=1.,
#                     upsampler='pixelshuffle',resi_connection='1conv',num_frames=3).to(device)

# model = PSRT(img_size=128, patch_size=1, in_chans=4, embed_dim=96, 
#             depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#             mlp_ratio=2, qkv_bias=True, qk_scale=False, 
#             drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
#             ape=False, patch_norm=True, use_checkpoint=False, 
#             upscale=1, img_range=1., upsampler='pixelshuffle', resi_connection='1conv', num_frames=3).to(device)

# model = PSRT_CCMS2(stage=2, 
#                     convnext_block_c=3,convnext_block_m=3,img_size=128,patch_size=1,
#                     in_chans=4,embed_dim=96,
#                     depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#                     mlp_ratio=2.,qkv_bias=True,qk_scale=None,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.1,
#                     norm_layer=nn.LayerNorm,ape=False,patch_norm=True,use_checkpoint=False,upscale=1,img_range=1.,
#                     upsampler='pixelshuffle',resi_connection='1conv',num_frames=3).to(device)

# model = PSRT_CCMS3(stage=2, 
#                     convnext_block_c=3,convnext_block_m=3,img_size=128,patch_size=1,
#                     in_chans=4,embed_dim=96,
#                     depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#                     mlp_ratio=2.,qkv_bias=True,qk_scale=None,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.1,
#                     norm_layer=nn.LayerNorm,ape=False,patch_norm=True,use_checkpoint=False,upscale=1,img_range=1.,
#                     upsampler='pixelshuffle',resi_connection='1conv',num_frames=3).to(device)

# model = PSRT_CCMS4(stage=2, 
#                     convnext_block_c=3,convnext_block_m=3,img_size=128,patch_size=1,
#                     in_chans=4,embed_dim=96,
#                     depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#                     mlp_ratio=2.,qkv_bias=True,qk_scale=None,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.1,
#                     norm_layer=nn.LayerNorm,ape=False,patch_norm=True,use_checkpoint=False,upscale=1,img_range=1.,
#                     upsampler='pixelshuffle',resi_connection='1conv',num_frames=3).to(device)

# model = PSRT_CCMS5(stage=2, 
#                     convnext_block_c=[1,1,1],convnext_block_m=[1,1,1],img_size=128,patch_size=1,
#                     in_chans=4,embed_dim=96,
#                     depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#                     mlp_ratio=2.,qkv_bias=True,qk_scale=None,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.1,
#                     norm_layer=nn.LayerNorm,ape=False,patch_norm=True,use_checkpoint=False,upscale=1,img_range=1.,
#                     upsampler='pixelshuffle',resi_connection='1conv',num_frames=3).to(device)

# model = PSRT_CCMS6_Vis(stage=2, 
#                     convnext_block_c=3,convnext_block_m=3,img_size=128,patch_size=1,
#                     in_chans=4,embed_dim=96,
#                     depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#                     mlp_ratio=2.,qkv_bias=True,qk_scale=None,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.1,
#                     norm_layer=nn.LayerNorm,ape=False,patch_norm=True,use_checkpoint=False,upscale=1,img_range=1.,
#                     upsampler='pixelshuffle',resi_connection='1conv',num_frames=3).to(device)


# model = PSRT_CCMS7_Vis(stage=2, 
#                     convnext_block_c=[1,1,1],convnext_block_m=[1,1,1],img_size=128,patch_size=1,
#                     in_chans=4,embed_dim=96,
#                     depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#                     mlp_ratio=2.,qkv_bias=True,qk_scale=None,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.1,
#                     norm_layer=nn.LayerNorm,ape=False,patch_norm=True,use_checkpoint=False,upscale=1,img_range=1.,
#                     upsampler='pixelshuffle',resi_connection='1conv',num_frames=3).to(device)

# model = PSRT_CCMS9_Vis(stage=2, 
#                     convnext_block_c=3,convnext_block_m=3,img_size=128,patch_size=1,
#                     in_chans=4,embed_dim=96,
#                     depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#                     mlp_ratio=2.,qkv_bias=True,qk_scale=None,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.1,
#                     norm_layer=nn.LayerNorm,ape=False,patch_norm=True,use_checkpoint=False,upscale=1,img_range=1.,
#                     upsampler='pixelshuffle',resi_connection='1conv',num_frames=3, kernel_size_k=9, kernel_size_n=1).to(device)

# model = PSRT_CCMS10_Vis(stage=2, 
#                     convnext_block_c=3,convnext_block_m=3,img_size=128,patch_size=1,
#                     in_chans=4,embed_dim=96,
#                     depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#                     mlp_ratio=2.,qkv_bias=True,qk_scale=None,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.1,
#                     norm_layer=nn.LayerNorm,ape=False,patch_norm=True,use_checkpoint=False,upscale=1,img_range=1.,
#                     upsampler='pixelshuffle',resi_connection='1conv',num_frames=3, kernel_size_k=31, kernel_size_n=31, fuse_type='add').to(device)

# model = PSRT_CCMS11_Vis(stage=2, 
#                     convnext_block_c=3,convnext_block_m=3,img_size=128,patch_size=1,
#                     in_chans=4,embed_dim=96,
#                     depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#                     mlp_ratio=2.,qkv_bias=True,qk_scale=None,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.1,
#                     norm_layer=nn.LayerNorm,ape=False,patch_norm=True,use_checkpoint=False,upscale=1,img_range=1.,
#                     upsampler='pixelshuffle',resi_connection='1conv',num_frames=3, kernel_size_k=15, kernel_size_n=15, fuse_type='add').to(device)

# model = PSRT_CCMS12_Vis(stage=2, 
#                     convnext_block_c=3,convnext_block_m=3,img_size=128,patch_size=1,
#                     in_chans=4,embed_dim=96,
#                     depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#                     mlp_ratio=2.,qkv_bias=True,qk_scale=None,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.1,
#                     norm_layer=nn.LayerNorm,ape=False,patch_norm=True,use_checkpoint=False,upscale=1,img_range=1.,
#                     upsampler='pixelshuffle',resi_connection='1conv',num_frames=3, fuse_type='add', 
#                     num_heads_s2d=4, num_blocks_s2d=[2,2,2,2,2,2,2], window_size_s2d=8).to(device)

# model = PSRT_CCMS13_Vis(stage=2, 
#                     convnext_block_c=3,convnext_block_m=3,img_size=128,patch_size=1,
#                     in_chans=4,embed_dim=96,
#                     depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#                     mlp_ratio=2.,qkv_bias=True,qk_scale=None,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.1,
#                     norm_layer=nn.LayerNorm,ape=False,patch_norm=True,use_checkpoint=False,upscale=1,img_range=1.,
#                     upsampler='pixelshuffle',resi_connection='1conv',num_frames=3, kernel_size_k=31, kernel_size_n=3, fuse_type='add').to(device)

# model = PSRT_CCMS14_Vis(stage=2, 
#                     convnext_block_c=3,convnext_block_m=3,img_size=128,patch_size=1,
#                     in_chans=4,embed_dim=96,
#                     depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#                     mlp_ratio=2.,qkv_bias=True,qk_scale=None,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.1,
#                     norm_layer=nn.LayerNorm,ape=False,patch_norm=True,use_checkpoint=False,upscale=1,img_range=1.,
#                     upsampler='pixelshuffle',resi_connection='1conv',num_frames=3, 
#                     dim_s2d= 96,stripe_size_s2d= 8,num_heads_s2d= 4,num_blocks_s2d= [2,2,2,2,2,2,2],
#                     mlp_ratio_s2d= 2,qkv_bias_s2d= True,drop_s2d= 0.0,attn_drop_s2d= 0.0,drop_path_s2d= 0.0,
#                     ).to(device)

# model = PSRT_CCMS16_Vis(stage=1, img_size=128, patch_size=1, in_chans=4, embed_dim=96, 
#             depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#             mlp_ratio=2, qkv_bias=True, qk_scale=False, 
#             drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
#             ape=False, patch_norm=True, use_checkpoint=False, 
#             upscale=1, img_range=1., upsampler='pixelshuffle', resi_connection='1conv', num_frames=3).to(device)

# model = PSRT_CCMS17_Vis(stage=1, img_size=128, patch_size=1, in_chans=4, embed_dim=96, 
#             depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#             mlp_ratio=2, qkv_bias=True, qk_scale=False, 
#             drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
#             ape=False, patch_norm=True, use_checkpoint=False, upscale=1, img_range=1., 
#             upsampler='pixelshuffle', resi_connection='1conv', num_frames=3, cnvxt_ker_size=(1,7,7)).to(device)

# model = PSRT(img_size=128, patch_size=1, in_chans=4, embed_dim=96, 
#                     depths=[6,6,6,6], num_heads=[6,6,6,6], window_size=[2,7,7], 
#                     mlp_ratio=2, qkv_bias=True, qk_scale=False, 
#                     drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
#                     ape=False, patch_norm=True, use_checkpoint=False, 
#                     upscale=1, img_range=1., upsampler='pixelshuffle', resi_connection='1conv', num_frames=3).to(device)

# model = PSRT_CCMS18_Vis(stage=2, img_size=128, patch_size=1, in_chans=4, embed_dim=96, 
#             depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#             mlp_ratio=2, qkv_bias=True, qk_scale=False, 
#             drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
#             ape=False, patch_norm=True, use_checkpoint=False, upscale=1, img_range=1., 
#             upsampler='pixelshuffle', resi_connection='1conv', num_frames=3).to(device)

# model = PSRT_CCMS7_3_Vis(stage=2, img_size=128, patch_size=1, in_chans=4, embed_dim=96, 
#             depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#             cdm_depths=[2,2,2], cdm_num_heads=[2,2,2], mpm_depths=[2,2,2], mpm_num_heads=[2,2,2],mpdm_dim=96,
#             mlp_ratio=2, qkv_bias=True, qk_scale=False, 
#             drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
#             ape=False, patch_norm=True, use_checkpoint=False, upscale=1, img_range=1., 
#             upsampler='pixelshuffle', resi_connection='1conv', num_frames=3).to(device)

# model = PSRT_CCMS7_4_Vis(stage=2, img_size=128, patch_size=1, in_chans=4, embed_dim=96, 
#             depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#             convnext_block_c=[1,1,1], convnext_block_m=[1,1,1],
#             mlp_ratio=2, qkv_bias=True, qk_scale=False, 
#             drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
#             ape=False, patch_norm=True, use_checkpoint=False, upscale=1, img_range=1., 
#             upsampler='pixelshuffle', resi_connection='1conv', num_frames=3).to(device)

# model = PSRT2(img_size=128, patch_size=1, in_chans=4, embed_dim=96, 
#                     depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#                     mlp_ratio=2, qkv_bias=True, qk_scale=False, 
#                     drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
#                     ape=False, patch_norm=True, use_checkpoint=False, 
#                     upscale=1, img_range=1., upsampler='pixelshuffle', resi_connection='1conv', num_frames=3,
#                     res_depths=[1,1,1], ffn_ratio=2).to(device)

# model = PSRT3(img_size=128, patch_size=1, in_chans=4, embed_dim=96, 
#                     depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#                     mlp_ratio=2, qkv_bias=True, qk_scale=False, 
#                     drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
#                     ape=False, patch_norm=True, use_checkpoint=False, 
#                     upscale=1, img_range=1., upsampler='pixelshuffle', resi_connection='1conv', num_frames=3).to(device)

# model = PSRT_CCMS6_4_Vis(stage=2, img_size=128,patch_size=1, in_chans=4,embed_dim=96,
#                     depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#                     mlp_ratio=2.,qkv_bias=True,qk_scale=None,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.1,
#                     norm_layer=nn.LayerNorm,ape=False,patch_norm=True,use_checkpoint=False,upscale=1,img_range=1.,
#                     upsampler='pixelshuffle',resi_connection='1conv',num_frames=3).to(device)

# model = PSRT_CCMS6_6_Vis(stage=1, img_size=128,patch_size=1, in_chans=4,embed_dim=96,
#                     depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#                     mlp_ratio=2.,qkv_bias=True,qk_scale=None,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.1,
#                     norm_layer=nn.LayerNorm,ape=False,patch_norm=True,use_checkpoint=False,upscale=1,img_range=1.,
#                     upsampler='pixelshuffle',resi_connection='1conv',num_frames=3).to(device)


# model = PSRT_CCMS6_7_Vis(stage=1, img_size=128,patch_size=1, in_chans=4,embed_dim=96,
#                     depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#                     mlp_ratio=2.,qkv_bias=True,qk_scale=None,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.1,
#                     norm_layer=nn.LayerNorm,ape=False,patch_norm=True,use_checkpoint=False,upscale=1,img_range=1.,
#                     upsampler='pixelshuffle',resi_connection='1conv',num_frames=3).to(device)

# model = PSRT_CCMS6_8_Vis(stage=1, img_size=128,patch_size=1, in_chans=4,embed_dim=96,
#                     depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#                     mlp_ratio=2.,qkv_bias=True,qk_scale=None,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.1,
#                     norm_layer=nn.LayerNorm,ape=False,patch_norm=True,use_checkpoint=False,upscale=1,img_range=1.,
#                     upsampler='pixelshuffle',resi_connection='1conv',num_frames=3, num_flow=3).to(device)

# model = PSRT_CCMS6_9_Vis(stage=1, img_size=128,patch_size=1, in_chans=4,embed_dim=96,
#                     depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#                     mlp_ratio=2.,qkv_bias=True,qk_scale=None,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.1,
#                     norm_layer=nn.LayerNorm,ape=False,patch_norm=True,use_checkpoint=False,upscale=1,img_range=1.,
#                     upsampler='pixelshuffle',resi_connection='1conv',num_frames=3,
#                     spynet_path= 'pretrained_model/spynet_sintel_final-3d2a1287.pth').to(device)

# model = PSRT4(img_size=128, patch_size=1, in_chans=4, embed_dim=96, 
#                     depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#                     mlp_ratio=2, qkv_bias=True, qk_scale=False, 
#                     drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
#                     ape=False, patch_norm=True, use_checkpoint=False, 
#                     upscale=1, img_range=1., upsampler='pixelshuffle', resi_connection='1conv', num_frames=3,
#                     res_depths=[1,1,1], ffn_ratio=2).to(device)

# model = PSRT7(img_size=128, patch_size=1, in_chans=4, embed_dim=96, 
#                     depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#                     mlp_ratio=2, qkv_bias=True, qk_scale=False, 
#                     drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
#                     ape=False, patch_norm=True, use_checkpoint=False, 
#                     upscale=1, img_range=1., upsampler='pixelshuffle', resi_connection='1conv', num_frames=3,
#                     res_depths=[1,1,1], ffn_ratio=2).to(device)

# model = PSRT(img_size=128, patch_size=1, in_chans=4, embed_dim=96, 
#                     depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#                     mlp_ratio=2, qkv_bias=True, qk_scale=False, 
#                     drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
#                     ape=False, patch_norm=True, use_checkpoint=False, 
#                     upscale=1, img_range=1., upsampler='pixelshuffle', resi_connection='1conv', num_frames=3).to(device)

# model = PSRT8(img_size=128, patch_size=1, in_chans=4, embed_dim=96, 
#                     depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#                     mlp_ratio=2, qkv_bias=True, qk_scale=False, 
#                     drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
#                     ape=False, patch_norm=True, use_checkpoint=False, 
#                     upscale=1, img_range=1., upsampler='pixelshuffle', resi_connection='1conv', num_frames=3).to(device)

# model = PSRT9(img_size=128, patch_size=1, in_chans=4, embed_dim=96, 
#                     depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#                     mlp_ratio=2, qkv_bias=True, qk_scale=False, 
#                     drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
#                     ape=False, patch_norm=True, use_checkpoint=False, 
#                     upscale=1, img_range=1., upsampler='pixelshuffle', resi_connection='1conv', num_frames=3).to(device)


# model = MoCHAformer(stage=1, convnext_block_c=3,convnext_block_m=3, img_size=128,patch_size=1, in_chans=4,embed_dim=96,
#                     depths=[5,5,5,5], num_heads=[4,4,4,4], window_size=[2,7,7], 
#                     mlp_ratio=2.,qkv_bias=True,qk_scale=None,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.1,
#                     norm_layer=nn.LayerNorm,ape=False,patch_norm=True,use_checkpoint=False,upscale=1,img_range=1.,
#                     upsampler='pixelshuffle',resi_connection='1conv',num_frames=3).to(device)

# model = MoCHAformer(stage=2, convnext_block_c=3,convnext_block_m=3, img_size=128,patch_size=1, in_chans=4,embed_dim=96,
#                     depths=[5,5,5,5], num_heads=[4,4,4,4], window_size=[2,7,7], 
#                     mlp_ratio=2.,qkv_bias=True,qk_scale=None,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.1,
#                     norm_layer=nn.LayerNorm,ape=False,patch_norm=True,use_checkpoint=False,upscale=1,img_range=1.,
#                     upsampler='pixelshuffle',resi_connection='1conv',num_frames=3).to(device)

model = MoCHAformer4(stage=2, 
                    convnext_block_c=3,convnext_block_m=3,img_size=128,patch_size=1,
                    in_chans=4,embed_dim=96,
                    depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
                    mlp_ratio=2.,qkv_bias=True,qk_scale=None,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.1,
                    norm_layer=nn.LayerNorm,ape=False,patch_norm=True,use_checkpoint=False,upscale=1,img_range=1.,
                    upsampler='pixelshuffle',resi_connection='1conv',num_frames=3).to(device)

# model = MoCHAformer4_ab4_2(stage=2, 
#                     convnext_block_c=3,convnext_block_m=3,img_size=128,patch_size=1,
#                     in_chans=4,embed_dim=96,
#                     depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#                     mlp_ratio=2.,qkv_bias=True,qk_scale=None,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.1,
#                     norm_layer=nn.LayerNorm,ape=False,patch_norm=True,use_checkpoint=False,upscale=1,img_range=1.,
#                     upsampler='pixelshuffle',resi_connection='1conv',num_frames=3, num_flow=3).to(device)

# model = MoCHAformer4_ab4_3(stage=2, 
#                     convnext_block_c=3,convnext_block_m=3,img_size=128,patch_size=1,
#                     in_chans=4,embed_dim=96,
#                     depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#                     mlp_ratio=2.,qkv_bias=True,qk_scale=None,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.1,
#                     norm_layer=nn.LayerNorm,ape=False,patch_norm=True,use_checkpoint=False,upscale=1,img_range=1.,
#                     upsampler='pixelshuffle',resi_connection='1conv',num_frames=3).to(device)

# model = MoCHAformer4_ab4_4(stage=2, 
#                     convnext_block_c=3,convnext_block_m=3,img_size=128,patch_size=1,
#                     in_chans=4,embed_dim=96,
#                     depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#                     mlp_ratio=2.,qkv_bias=True,qk_scale=None,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.1,
#                     norm_layer=nn.LayerNorm,ape=False,patch_norm=True,use_checkpoint=False,upscale=1,img_range=1.,
#                     upsampler='pixelshuffle',resi_connection='1conv',num_frames=3).to(device)

# model = MoCHAformer4_ab4_5(stage=2, 
#                     convnext_block_c=3,convnext_block_m=3,img_size=128,patch_size=1,
#                     in_chans=4,embed_dim=96,
#                     depths=[3,3,3], num_heads=[4,4,4], window_size=[2,7,7], 
#                     mlp_ratio=2.,qkv_bias=True,qk_scale=None,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.1,
#                     norm_layer=nn.LayerNorm,ape=False,patch_norm=True,use_checkpoint=False,upscale=1,img_range=1.,
#                     upsampler='pixelshuffle',resi_connection='1conv',num_frames=3,
#                     spynet_path="pretrained_model/spynet_sintel_final-3d2a1287.pth").to(device)

# #############################################################
# input_rgb = torch.randn(1, 3, 3, 720, 1280).to(device)
# input_raw = torch.randn(1, 3, 4, 360, 640).to(device)
# input_flow = torch.randn(1, 2, 2, 720, 1280).to(device)

# input_rgb_img = torch.randn(1, 3, 720, 1280).to(device)
# #############################################################
# input_rgb = torch.randn(1, 3, 3, 360, 640).to(device)
# input_raw = torch.randn(1, 3, 4, 180, 320).to(device)
# input_flow = torch.randn(1, 2, 2, 360, 640).to(device)

# input_rgb_img = torch.randn(1, 3, 360, 640).to(device)

# #############################################################
input_rgb = torch.randn(1, 3, 3, 180, 320).to(device)
input_raw = torch.randn(1, 3, 4, 90, 160).to(device)
input_flow = torch.randn(1, 2, 2, 180, 320).to(device)

input_rgb_img = torch.randn(1, 3, 180, 320).to(device)

def compleity(batch_size):
    flops_g, macs_g, params_g = get_model_profile(model=model,  # model
                                                    input_shape=None,
                                                    # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
                                                    # args=None,  # list of positional arguments to the model.
                                                    # # multi branch input
                                                    # args=[input_rgb, input_raw, 'train'],
                                                    # single branch input
                                                    # args=[input_rgb, input_raw],
                                                    args=[input_raw],
                                                    # args = [input_rgb_img],
                                                    kwargs=None,  # dictionary of keyword arguments to the model.
                                                    print_profile=True,
                                                    # prints the model graph with the measured profile attached to each module
                                                    detailed=True,  # print the detailed profile
                                                    module_depth=-1,
                                                    # depth into the nested modules, with -1 being the inner most modules
                                                    top_modules=1,
                                                    # the number of top modules to print aggregated profile
                                                    warm_up=1,
                                                    # the number of warm-ups before measuring the time of each module
                                                    as_string=True,
                                                    # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                                    output_file=None,
                                                    # path to the output file. If None, the profiler prints to stdout.
                                                    ignore_modules=None)  # the list of modules to ignore in the profiling
    return flops_g, macs_g, params_g

compleity(1)

input_rgb_for_inference = torch.randn(1, 3, 3, 720, 1280).to(device)
input_raw_for_inference = torch.randn(1, 3, 4, 360, 640).to(device)
input_flow_for_inference = torch.randn(1, 2, 2, 720, 1280).to(device)


num_runs = 100
total_time = 0

for _ in range(num_runs):
    start_time = time.time()
    with torch.no_grad():
        phase = 'test'
        output = model(input_raw_for_inference)
        # output = model(input_rgb_for_inference, input_raw_for_inference, phase, is_train=False)
    end_time = time.time()
    total_time += (end_time - start_time)

average_inference_time = total_time / num_runs
print(f"Average inference time over {num_runs} runs: {average_inference_time:.6f} seconds")

"""
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python vd/utils/MAC_Gflops_params.py
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=1 python vd/utils/MAC_Gflops_params.py
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=2 python vd/utils/MAC_Gflops_params.py
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=3 python vd/utils/MAC_Gflops_params.py
"""
