import torch
import os.path as osp
import time
from tqdm import tqdm
from collections import OrderedDict
import re
import copy
from logging import getLogger
from copy import deepcopy
from einops import rearrange

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.archs import build_network
from basicsr.models.sr_model import SRModel
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger
from basicsr.losses import build_loss
from basicsr.metrics.psnr_ssim import calculate_ssim_pt,calculate_psnr_pt
from basicsr.metrics.lpips import calculate_lpips_pt

from vd.metrics.vd_metric import calculate_vd_ssim, calculate_vd_psnr
from vd.data.data_util import tensor2numpy, imwrite_gt

from deepspeed.profiling.flops_profiler import get_model_profile
from vd.utils.test_visualize import store_vis_data

@MODEL_REGISTRY.register()
class MoCHAformer_stg2_model(SRModel):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define networks
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # self.net_f = build_network(self.opt['network_f'])
        # self.net_f = self.model_to_device(self.net_f)
        # self.print_network(self.net_f)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        # load pretrained models
        # load_path_f = self.opt['path'].get('pretrain_network_f', None)
        # if load_path_f is not None:
        #     param_key_f = self.opt['path'].get('param_key_f', 'params')
        #     self.load_network(self.net_f, load_path_f, self.opt['path'].get('strict_load_f', True), param_key_f)

        if self.is_train:
            self.init_training_settings()

    # override the SR_Model's setup_optimizers function
    # ----------------------------------------------------------------------
    def setup_optimizers(self):
        def _make_prefix_set(raw_added):
            """added_net 항목을 문자열 · 리스트 · 튜플 · 세트 등 어떤 형태로 주어도
            ‘경로 prefix’ 집합(set) 으로 변환한다."""
            if isinstance(raw_added, str):
                # 콤마로 구분된 단일 문자열일 가능성
                parts = [s.strip() for s in raw_added.split(",")]
            elif isinstance(raw_added, (list, tuple, set)):
                parts = list(raw_added)
            else:
                parts = []
            # 빈 문자열 제거
            return {p for p in parts if p}

        def _is_under_prefix(name: str, prefixes):
            """파라미터 이름이 prefixes 중 하나로 시작하는지 판단.
            • 정확히 일치하거나
            • 'prefix.' 로 이어지는 경우 둘 다 True."""
            for p in prefixes:
                if name == p or name.startswith(p + "."):
                    return True
            return False
        
        """
        ① self.opt['network_g']['added_net'] 에 명시된 '모듈 경로 prefix' 아래에
        존재하는 모든 파라미터를 'new_params' 로 분류
        ② 나머지는 'base_params'
        ③ 두 그룹에 대해 서로 다른 Optimizer 생성
        """
        logger = get_root_logger()
        train_opt = self.opt["train"]

        # 1. added_net 파싱 → prefix 집합
        raw_added = self.opt.get("network_g", {}).get("added_net", [])
        added_prefixes = _make_prefix_set(raw_added)

        # 2. 파라미터 분류
        base_params, new_params = [], []
        # new_name = []
        for name, param in self.net_g.named_parameters():
            if not param.requires_grad:
                logger.warning(f"Params {name} will not be optimized.")
                continue
            if _is_under_prefix(name, added_prefixes):
                new_params.append(param)
                # new_name.append(name)
            else:
                base_params.append(param)

        # 3-1. 기존 파라미터용 Optimizer
        base_cfg  = deepcopy(train_opt["optim_g"])
        base_type = base_cfg.pop("type")
        self.optimizer_g = self.get_optimizer(base_type, base_params, **base_cfg)
        self.optimizers.append(self.optimizer_g)

        # 3-2. 신규 파라미터용 Optimizer
        if "optim_d" in train_opt:
            new_cfg = deepcopy(train_opt["optim_d"])
        else:
            new_cfg = deepcopy(train_opt["optim_d"])
        new_type = new_cfg.pop("type")
        self.optimizer_d = self.get_optimizer(new_type, new_params, **new_cfg)
        self.optimizers.append(self.optimizer_d)

    def init_training_settings(self):
        self.net_g.train()
        # self.net_f.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # self.net_f_ema = build_network(self.opt['network_f']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define l1 loss
        if train_opt.get('pixel_l1_opt'):
            self.cri_l1_pix = build_loss(train_opt['pixel_l1_opt']).to(self.device)
        else:
            self.cri_l1_pix = None

        # define perceptual loss
        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None
        
        # define l2 loss for coarse clean frames
        if train_opt.get('pixel_l2_ccf_opt'):
            self.cri_l2_ccf_pix = build_loss(train_opt['pixel_l2_ccf_opt']).to(self.device)
        else:
            self.cri_l2_ccf_pix = None
        
        # define l2 loss for self-reconstruction supervision
        if train_opt.get('pixel_l2_srs_opt'):
            self.cri_l2_srs_pix = build_loss(train_opt['pixel_l2_srs_opt']).to(self.device)
        else:
            self.cri_l2_srs_pix = None
        
        # low rank approximation loss / == WNNM loss
        if train_opt.get('wnnm_opt'):
            self.cri_wnnm = build_loss(train_opt['wnnm_opt']).to(self.device)
        else:
            self.cri_wnnm = None
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
    
    def optimize_parameters(self, current_iter):
        b, t, c, h, w = self.lq.shape
        self.optimizer_g.zero_grad()
        self.optimizer_d.zero_grad()
        self.output, self.moire_pat, self.coarse_clean_raw, self.coarse_clean_rgb, self.out_vis = self.net_g(self.lq, phase='train')
        # self.output = self.net_f(self.output1, align_feat)

        l_total = 0
        loss_dict = OrderedDict()
        
        # pixel L1 loss
        if self.cri_l1_pix:
            l_l1_pix = self.cri_l1_pix(self.output, self.gt[:,t//2,:,:,:])
            l_total += l_l1_pix
            loss_dict['l_l1_pix'] = l_l1_pix
        
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt[:,t//2,:,:,:])
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        
        
        # Low rank approximation loss (for moire pattern constraint)
        if self.cri_wnnm:
            l_wnnm_sum = 0.0
            for i in range(t):
                loss_i = self.cri_wnnm(self.moire_pat[:, i, ...])  # [B,C,H,W]
                l_wnnm_sum += loss_i
            l_wnnm_demoire = l_wnnm_sum / t          # 프레임 평균
            l_total += l_wnnm_demoire
            loss_dict['l_wnnm_demoire'] = l_wnnm_demoire
                
        # pixel L2 loss: for coarse clean frames
        if self.cri_l2_ccf_pix:
            l_l2_ccf_pix = self.cri_l2_ccf_pix(self.coarse_clean_rgb, self.gt)
            l_total += l_l2_ccf_pix
            loss_dict['l_l2_ccf_pix'] = l_l2_ccf_pix
        
        # self-reconstruction loss         
        if self.cri_l2_srs_pix:
            l_l2_srs_pix = self.cri_l2_srs_pix((self.coarse_clean_raw + self.moire_pat), self.lq)
            l_total += l_l2_srs_pix
            loss_dict['l_l2_srs_pix'] = l_l2_srs_pix
         
        l_total.backward()
        self.optimizer_g.step()
        self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        scale = self.opt.get('scale', 1)
        _, _, _, h_old, w_old = self.gt.size()

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            # self.net_f.eval()
            with torch.no_grad():
                self.output, self.moire_pat, self.coarse_clean_raw, self.coarse_clean_rgb, self.out_vis = self.net_g_ema(self.lq, phase='val', is_train=self.opt['is_train'])
                # self.output = self.output[:, :, :h_old * scale, :w_old * scale]
                # self.output = self.net_f(self.output1, align_feat)
                # self.output = self.output[:, :, :h_old * scale, :w_old * scale]
        else:
            self.net_g.eval()
            # self.net_f.eval()
            with torch.no_grad():
                self.output, self.moire_pat, self.coarse_clean_raw, self.coarse_clean_rgb, self.out_vis = self.net_g(self.lq, phase='val', is_train=self.opt['is_train'])
                # self.output = self.output[:, :, :h_old * scale, :w_old * scale]
                # self.output = self.net_f(self.output1, align_feat)
                # self.output = self.output[:, :, :h_old * scale, :w_old * scale]
            self.net_g.train()
            # self.net_f.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
                # self.metric_results['SSIM_inter'] = 0
                # self.metric_results['PSNR_inter'] = 0
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        time_inf_total = 0.
        for idx, val_data in enumerate(dataloader):
            img_name = val_data['key'][0]
            self.feed_data(val_data)
            st = time.time()
            self.test()
            st1 = time.time() - st
            # print('The test time for this image %.3f' % st1)
            time_inf_total += st1

            visuals = self.get_current_visuals()
            if self.opt['is_train']:
                sr_img_tensors = self.output.detach()
                metric_data['img'] = sr_img_tensors
                if 'gt' in visuals:
                    b,t,c,h,w = self.gt.shape
                    gt_img_tensors = self.gt[:,t//2,:,:,:].detach()
                    metric_data['img2'] = gt_img_tensors
                    del self.gt
            else:
                sr_img = tensor2numpy(visuals['result'])
                metric_data['img'] = sr_img
                if 'gt' in visuals:
                    b, t, c, h, w = self.gt.shape
                    _, _, c2, _, _ = self.lq.shape
                    
                    # self.out_vis['01_gt_rgb'] = visuals['gt']
                    store_vis_data(self.out_vis, '01_gt_rgb', rearrange(self.gt, 'b t c h w -> b c t h w', b=b, c=c, t=t), 'rgb', num_flow=1)
                    store_vis_data(self.out_vis, '01_coarse_clean_raw', rearrange(self.coarse_clean_raw, 'b t c h w -> b c t h w', b=b, c=c2, t=t), 'raw', num_flow=1)
                    store_vis_data(self.out_vis, '01_coarse_clean_rgb', rearrange(self.coarse_clean_rgb, 'b t c h w -> b c t h w', b=b, c=c, t=t), 'rgb', num_flow=1)
                    store_vis_data(self.out_vis, '01_moire_pat_raw', rearrange(self.moire_pat, 'b t c h w -> b c t h w', b=b, c=c2, t=t), 'raw', num_flow=1)
                    store_vis_data(self.out_vis, '02_pred_clean_rgb', self.output, 'rgb', num_flow=1)

                    gt_img = tensor2numpy(visuals['gt'][:,t//2,:,:,:])
                    metric_data['img2'] = gt_img
                    del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            del self.moire_pat
            del self.coarse_clean_raw
            del self.coarse_clean_rgb
            
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    pass
                else:
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                             f'{img_name}.png')
                    imwrite_gt(sr_img, save_img_path)
                    # save tensor (of visualization results)
                    save_pth_path = osp.join(self.opt['path']['visualization'], f'{img_name}_tensor.pth')
                    with open(save_pth_path, 'wb') as f:
                        torch.save(self.out_vis, f)    

            if with_metrics:
                # if self.opt['is_train']:
                #     self.metric_results['SSIM_inter'] += calculate_ssim_pt(self.output1.detach(),
                #                                                            metric_data['img2'],
                #                                                            0).detach().cpu().numpy().sum()
                #     self.metric_results['PSNR_inter'] += calculate_psnr_pt(self.output1.detach(),
                #                                                            metric_data['img2'],
                #                                                            0).detach().cpu().numpy().sum()
                # else:
                #     self.metric_results['SSIM_inter'] += calculate_vd_ssim(sr_img_inter,
                #                                                            metric_data['img2']).sum()
                #     self.metric_results['PSNR_inter'] += calculate_vd_psnr(sr_img_inter,
                #                                                            metric_data['img2']).sum()
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    if self.opt['is_train']:
                        self.metric_results[name] += calculate_metric(metric_data, opt_).detach().cpu().numpy().sum()
                    else:
                        self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        time_avg = time_inf_total / (idx + 1)
        # print('The average test time is %.3f' % time_avg)
        if with_metrics:
            for metric in self.metric_results.keys():
                if self.opt['is_train']:
                    self.metric_results[metric] /= 3000
                else:
                    self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        # self.save_network(self.net_f, 'net_f', current_iter)
        self.save_training_state(epoch, current_iter)

    def compleity(self, batch_size):
        flops_g, macs_g, params_g = get_model_profile(model=self.net_g,  # model
                                                      input_shape=(batch_size, 3, 3, 720, 1280),
                                                      # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
                                                      args=None,  # list of positional arguments to the model.
                                                      kwargs=None,  # dictionary of keyword arguments to the model.
                                                      print_profile=True,
                                                      # prints the model graph with the measured profile attached to each module
                                                      detailed=True,  # print the detailed profile
                                                      module_depth=-1,
                                                      # depth into the nested modules, with -1 being the inner most modules
                                                      top_modules=1,
                                                      # the number of top modules to print aggregated profile
                                                      warm_up=10,
                                                      # the number of warm-ups before measuring the time of each module
                                                      as_string=True,
                                                      # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                                      output_file=None,
                                                      # path to the output file. If None, the profiler prints to stdout.
                                                      ignore_modules=None)  # the list of modules to ignore in the profiling

        flops_f, macs_f, params_f = get_model_profile(model=self.net_f,  # model
                                                      input_shape=None,
                                                      # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
                                                      args=[torch.zeros((1, 3, 720, 1280), device=self.device),
                                                            torch.zeros((1, 216, 180, 320), device=self.device)],  # list of positional arguments to the model.
                                                      kwargs=None,  # dictionary of keyword arguments to the model.
                                                      print_profile=True,
                                                      # prints the model graph with the measured profile attached to each module
                                                      detailed=True,  # print the detailed profile
                                                      module_depth=-1,
                                                      # depth into the nested modules, with -1 being the inner most modules
                                                      top_modules=1,
                                                      # the number of top modules to print aggregated profile
                                                      warm_up=10,
                                                      # the number of warm-ups before measuring the time of each module
                                                      as_string=True,
                                                      # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                                      output_file=None,
                                                      # path to the output file. If None, the profiler prints to stdout.
                                                      ignore_modules=None)  # the list of modules to ignore in the profiling

        return flops_g, macs_g, params_g, flops_f, macs_f, params_f