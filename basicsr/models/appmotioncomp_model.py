import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm


from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img, make_coordinate_grid, show_feature_map, mimsave
from basicsr.utils.registry import MODEL_REGISTRY
import torch.nn.functional as F
from .sr_model import SRModel

from torch import autograd as autograd
import flow_vis
import numpy as np
from einops import rearrange
import imageio

from basicsr.utils.motion_estimator_util import TPS
from torch.nn.utils import clip_grad_norm_
import math

from scipy.spatial import ConvexHull
import cv2

# relative motion
def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new

class Transform:
    """
    Random tps transformation for equivariance constraints. See Sec 3.3
    """
    def __init__(self, bs, **kwargs):
        noise = torch.normal(mean=0, std=kwargs['sigma_affine'] * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs

        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid((kwargs['points_tps'], kwargs['points_tps']), type=noise.type())
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(mean=0,
                                               std=kwargs['sigma_tps'] * torch.ones([bs, 1, kwargs['points_tps'] ** 2]))
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = make_coordinate_grid(frame.shape[2:], type=frame.type()).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        return F.grid_sample(frame, grid, padding_mode="reflection")

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)

        if self.tps:
            control_points = self.control_points.type(coordinates.type())
            control_params = self.control_params.type(coordinates.type())
            distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
            distances = torch.abs(distances).sum(-1)

            result = distances ** 2
            result = result * torch.log(distances + 1e-6)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result
        return transformed

    def jacobian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        grad_x = autograd.grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        grad_y = autograd.grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        return jacobian
    def hessian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        grad_x = autograd.grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        grad_y = autograd.grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        return jacobian

# use vector quantization codebook, trained jointly
@MODEL_REGISTRY.register()
class AppMotionCompModel(SRModel):
    def feed_data(self, data):
        self.gt = data['driving'].to(self.device)
        self.source = data['source'].to(self.device)
        self.b = self.gt.shape[0]


    def init_training_settings(self):
        logger = get_root_logger()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        self.scale_adaptive_gan_weight = train_opt.get('scale_adaptive_gan_weight', 0.8)

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True))
        
        # define network motion estimator
        self.motion_estimator = build_network(self.opt['network_motion_estimator'])
        self.motion_estimator = self.model_to_device(self.motion_estimator)
        self.print_network(self.motion_estimator)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_motion_estimator', None)
        if load_path is not None:
            self.load_network(self.motion_estimator, load_path, self.opt['path'].get('strict_load_motion_estimator', True))

        logger.info(f"with_position_emb: {self.opt['network_g'].get('with_position_emb', False)}")

        self.net_g.train()
        self.net_d.train()
        self.motion_estimator.train()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)
        
        if train_opt.get('motion_codebook_code_opt'):
            self.l_weight_motion_codebook_code = train_opt['motion_codebook_code_opt'].get('loss_weight', 1.0)
        else:
            self.l_weight_motion_codebook_code = 1.0
        
        if train_opt.get('motion_codebook_recon_opt'):
            self.cri_motion_codebook_recon = build_loss(train_opt['motion_codebook_recon_opt']).to(self.device)
        else:
            self.cri_motion_codebook_recon = None
        
        if train_opt.get('app_codebook_code_opt'):
            self.l_weight_app_codebook_code = train_opt['app_codebook_code_opt'].get('loss_weight', 1.0)
        else:
            self.l_weight_app_codebook_code = 1.0

        logger.info(f'vqgan_quantizer_motion/app: {self.opt["network_g"]["quantizer_type"]}')
        
        if train_opt.get('equivariance_opt'):
            self.cri_equivariance = build_loss(train_opt['equivariance_opt']).to(self.device)
        else:
            self.cri_equivariance = None
        
        if train_opt.get('kp_distance_opt'):
            self.cri_kp_distancce = build_loss(train_opt['kp_distance_opt']).to(self.device)
        else:
            self.cri_kp_distancce = None
        
        if train_opt.get('lr_pixel_perceptual_opt'):
            self.l_weight_lr_pixel_perceptual_list = train_opt['lr_pixel_perceptual_opt'].get('loss_weight', [])
        else:
            self.l_weight_lr_pixel_perceptual_list = []
        
        self.fix_generator = train_opt.get('fix_generator', True)
        logger.info(f'fix_generator: {self.fix_generator}')

        self.net_g_start_iter = train_opt.get('net_g_start_iter', 0)
        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_start_iter = train_opt.get('net_d_start_iter', 0)

        self.disc_weight = train_opt.get('disc_weight', 0.8)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def calculate_adaptive_weight(self, recon_loss, g_loss, last_layer, disc_weight_max):
        recon_grads = torch.autograd.grad(recon_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(recon_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, disc_weight_max).detach()
        return d_weight
    
    def adopt_weight(self, weight, global_step, threshold=0, value=0.):
        if global_step < threshold:
            weight = value
        return weight

    def setup_optimizers(self):
        train_opt = self.opt['train']
        # optimizer g
        optim_params_g = []
        # net g
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params_g.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} in self.net_g will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params_g, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        # optimizer motion estimator
        # motion estimator
        self.optimizer_m = train_opt.get('optim_motion', None)
        if self.optimizer_m is not None:
            optim_params_m = []

            for k, v in self.motion_estimator.named_parameters():
                if v.requires_grad:
                    optim_params_m.append(v)
                else:
                    logger = get_root_logger()
                    logger.warning(f'Params {k} in self.motion_estimator will not be optimized.')
            
            optim_type = train_opt['optim_motion'].pop('type')
            self.optimizer_m = self.get_optimizer(optim_type, optim_params_m, **train_opt['optim_motion'])
            self.optimizers.append(self.optimizer_m)

        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def gray_resize_for_identity(self, out, size=128):
        out_gray = (0.2989 * out[:, 0, :, :] + 0.5870 * out[:, 1, :, :] + 0.1140 * out[:, 2, :, :])
        out_gray = out_gray.unsqueeze(1)
        out_gray = F.interpolate(out_gray, (size, size), mode='bilinear', align_corners=False)
        return out_gray

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear', align_corners=True)
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation,align_corners=True)

    def occlude_input(self, inp, occlusion_map):
        if inp.shape[2] != occlusion_map.shape[2] or inp.shape[3] != occlusion_map.shape[3]:
            occlusion_map = F.interpolate(occlusion_map, size=inp.shape[2:], mode='bilinear',align_corners=True)
        out = inp * occlusion_map
        return out
    
    def optimize_parameters(self, current_iter):
        logger = get_root_logger()

        # optimize net_g, motion_estimator
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        if self.optimizer_m is not None:
            self.optimizer_m.zero_grad()

        self.dense_motion = self.motion_estimator(self.gt, self.source)
        self.out_dict= self.net_g(self.source, self.dense_motion, w=1, gt=self.gt)
        
        l_g_total = 0
        loss_dict = OrderedDict()

        if current_iter % self.net_d_iters == 0 and current_iter > self.net_g_start_iter:
            # pixel loss 
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.out_dict['out'], self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix

            # perceptual loss
            if self.cri_perceptual:
                l_g_percep = self.cri_perceptual(self.out_dict['out'], self.gt)
                l_g_total += l_g_percep
                loss_dict['l_g_percep'] = l_g_percep

            # gan loss
            if  current_iter > self.net_d_start_iter:
                fake_g_pred = self.net_d(self.out_dict['out'])
                l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
                #recon_loss = l_g_pix + l_g_percep
                recon_loss = l_g_percep
                if self.cri_pix:
                    recon_loss += l_g_pix

                if not self.fix_generator:
                    last_layer = self.net_g.module.generator.blocks[-1].weight
                    d_weight = self.calculate_adaptive_weight(recon_loss, l_g_gan, last_layer, disc_weight_max=1.0)

                else:
                    largest_fuse_size = self.opt['network_g']['connect_list'][-1]
                    last_layer = self.net_g.module.fuse_convs_dict[largest_fuse_size].shift[-1].weight
                    d_weight = self.calculate_adaptive_weight(recon_loss, l_g_gan, last_layer, disc_weight_max=1.0)
                                        
                d_weight *= self.scale_adaptive_gan_weight # 0.8
                loss_dict['d_weight'] = d_weight
                l_g_total += d_weight * l_g_gan
                loss_dict['l_g_gan'] = d_weight * l_g_gan

            if self.l_weight_motion_codebook_code:
                l_g_motion_codebook_code = 0
                for l in self.out_dict['codebook_loss_motion_list']:
                    l_g_motion_codebook_code += l * self.l_weight_motion_codebook_code
                l_g_total += l_g_motion_codebook_code
                loss_dict['l_g_motion_codebook_code'] = l_g_motion_codebook_code
            
            if self.cri_motion_codebook_recon:
                xx = torch.linspace(-1., 1., self.out_dict['deformation_list'][0].shape[1])
                yy = torch.linspace(-1., 1., self.out_dict['deformation_list'][0].shape[2])
                grid_x, grid_y = torch.meshgrid(xx, yy, indexing='xy')
                grid = torch.cat((grid_x.unsqueeze(0).unsqueeze(-1), grid_y.unsqueeze(0).unsqueeze(-1)), dim=-1).to(self.out_dict['deformation_list'][0].device)
                
                l_g_motion_codebook_recon = 0
                for i in range(len(self.out_dict['motion_recon_list'])):
                    l_g_motion_codebook_recon += self.cri_motion_codebook_recon(self.out_dict['motion_recon_list'][i].permute(0,3,1,2), (self.out_dict['deformation_list'][i]-grid).permute(0,3,1,2).detach())

                l_g_total += l_g_motion_codebook_recon
                loss_dict['l_g_motion_codebook_recon'] = l_g_motion_codebook_recon
            
            if len(self.l_weight_lr_pixel_perceptual_list)>0:
                for i in range(len(self.l_weight_lr_pixel_perceptual_list)): # i=0
                    if self.cri_pix:
                        l_g_pix_lr = self.cri_pix(self.out_dict['out_lr'][i], self.gt) * self.l_weight_lr_pixel_perceptual_list[i]
                        l_g_total += l_g_pix_lr
                        loss_dict['l_g_pix_lr_'+str(i)] = l_g_pix_lr
                    
                    if self.cri_perceptual:
                        l_g_percep_lr = self.cri_perceptual(self.out_dict['out_lr'][i], self.gt) * self.l_weight_lr_pixel_perceptual_list[i]
                        l_g_total += l_g_percep_lr
                        loss_dict['l_g_percep_lr_'+str(i)] = l_g_percep_lr
            
            if self.l_weight_app_codebook_code>0:
                l_g_app_codebook_code = 0
                for l in self.out_dict['codebook_loss_app_list']:
                    l_g_app_codebook_code += l * self.l_weight_app_codebook_code
                l_g_total += l_g_app_codebook_code
                loss_dict['l_g_app_codebook_code'] = l_g_app_codebook_code


            if self.cri_equivariance:
                transform = Transform(self.gt.shape[0], **self.cri_equivariance.transform_params)
                transformed_frame = transform.transform_frame(self.gt)
                transformed_kp = self.motion_estimator.module.kp_detector(transformed_frame)

                l_equivariance_value, l_equivariance_jacobian = self.cri_equivariance(self.dense_motion['kp_driving'], transformed_kp, transform)
                l_g_total += l_equivariance_value
                loss_dict['l_equivariance_value'] = l_equivariance_value

                if l_equivariance_jacobian is not None:
                    l_g_total += l_equivariance_jacobian
                    loss_dict['l_equivariance_jacobian'] = l_equivariance_jacobian
            
            if self.cri_kp_distancce:
                l_kpd = self.cri_kp_distancce(kp_driving = self.dense_motion['kp_driving'], kp_source = self.dense_motion['kp_source'])
                l_g_total += l_kpd
                loss_dict['l_kpd'] = l_kpd
            
            l_g_total.backward()
            self.optimizer_g.step()
            if self.optimizer_m is not None:
                self.optimizer_m.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        # optimize net_d
        if current_iter > self.net_d_start_iter:
            for p in self.net_d.parameters():
                p.requires_grad = True

            self.optimizer_d.zero_grad()
            # real
            real_d_pred = self.net_d(self.gt)
            l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
            loss_dict['l_d_real'] = l_d_real
            loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
            l_d_real.backward()
            # fake
            fake_d_pred = self.net_d(self.out_dict['out'].detach())
            l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
            loss_dict['l_d_fake'] = l_d_fake
            loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
            l_d_fake.backward()

            self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)


    def test(self):
        with torch.no_grad():
            self.dense_motion = self.motion_estimator(self.gt, self.source)
            # use self.net_g
            logger = get_root_logger()

            self.net_g.eval()

            w = self.opt['val'].get('w', 1)
            draw_fig = self.opt.get('draw_fig', False)

            self.out_dict = self.net_g(self.source, self.dense_motion, w=w, visualize_app_feat=draw_fig, inference=True)

            self.driving_feat = self.net_g.encode_driving(self.gt)
            self.source_feat = self.net_g.encode_driving(self.source)

            if 'lq_feat' in self.out_dict:
                self.lq_recon = self.net_g.generator(self.out_dict['lq_feat'])

            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)


    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        pbar = tqdm(total=len(dataloader), unit='image')

        if self.opt.get('network_motion_estimator', None) is not None:
            self.motion_estimator = build_network(self.opt['network_motion_estimator']).to(self.device)
            self.motion_estimator.eval()
            self.print_network(self.motion_estimator)
            
            # load pretrained models
            load_path = self.opt['path'].get('pretrain_network_motion_estimator', None)
            if load_path is not None:
                param_key = self.opt['path'].get('param_key_m', 'params')
                self.load_network(self.motion_estimator, load_path, self.opt['path'].get('strict_load_motion_estimator', True), param_key)

        draw_fig = self.opt.get('draw_fig', False)
                
        for idx, val_data in enumerate(dataloader):
            img_name = val_data['frame_name'][0]
            self.feed_data(val_data)
            self.test() 

            visuals = self.get_current_visuals()

            result_img = tensor2img([visuals['result']], rgb2bgr=True, min_max=(-1, 1))

            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=True, min_max=(-1, 1))
                if not draw_fig:
                    del self.gt

                visual = tensor2img([torch.cat((visuals['source'],visuals['gt'], visuals['result']),3)], rgb2bgr=True, min_max=(-1, 1))

                if 'recon' in visuals:
                    visual = np.concatenate((visual, tensor2img(visuals['recon'], rgb2bgr=True, min_max=(-1, 1))), axis=1)

                source = tensor2img([visuals['source']], rgb2bgr=True, min_max=(-1, 1))

            # tentative for out of GPU memory
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    imwrite(visual, osp.join(self.opt['path']['visualization'], dataset_name, 'visual', f'{img_name}_v.png'))
                    imwrite(result_img, osp.join(self.opt['path']['visualization'], dataset_name, 'result', f'{img_name}_r.png'))
                    imwrite(source, osp.join(self.opt['path']['visualization'], dataset_name, 'source', f'{img_name}_s.png'))
                    imwrite(gt_img, osp.join(self.opt['path']['visualization'], dataset_name, 'driving', f'{img_name}_d.png'))
                    
            
            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    if name in ['psnr', 'ssim', 'l1']:
                        metric_data = dict(img1=result_img, img2=gt_img)
                        self.metric_results[name] += calculate_metric(metric_data, opt_)
                        
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric, opt_ in self.opt['val']['metrics'].items():
                if metric in ['psnr', 'ssim', 'l1']:
                    self.metric_results[metric] /= (idx + 1)
                    if metric in ['l1']:
                        self.metric_results.update({'l1_255': self.metric_results['l1']/255.})
                    print(metric, self.metric_results[metric])

                elif metric in ['fid']:
                    metric_data = dict(paths=[osp.join(self.opt['path']['visualization'], dataset_name, 'source'), osp.join(self.opt['path']['visualization'], dataset_name, 'result')])
                    self.metric_results[metric] = calculate_metric(metric_data, opt_)

                elif metric in ['lpips']:
                    metric_data = dict(path1=osp.join(self.opt['path']['visualization'], dataset_name, 'result'), path2=osp.join(self.opt['path']['visualization'], dataset_name, 'driving'))
                    self.metric_results[metric] = calculate_metric(metric_data, opt_)
                
                elif metric in ['face_akd']:
                    metric_data = dict(path_gt=osp.join(self.opt['path']['visualization'], dataset_name, 'driving'), path_generated=osp.join(self.opt['path']['visualization'], dataset_name, 'result'))
                    self.metric_results[metric] = calculate_metric(metric_data, opt_)
                
                elif metric in ['face_aed']:
                    if self.opt['val']['cross_id']:
                        metric_data = dict(path_gt=osp.join(self.opt['path']['visualization'], dataset_name, 'source'), path_generated=osp.join(self.opt['path']['visualization'], dataset_name, 'result'))
                    else:
                        metric_data = dict(path_gt=osp.join(self.opt['path']['visualization'], dataset_name, 'driving'), path_generated=osp.join(self.opt['path']['visualization'], dataset_name, 'result'))
                    
                    self.metric_results[metric] = calculate_metric(metric_data, opt_)
                
                elif metric in ['id_similarity']:
                    if self.opt['val']['cross_id']:
                        metric_data = dict(path_gt=osp.join(self.opt['path']['visualization'], dataset_name, 'source'), path_generated=osp.join(self.opt['path']['visualization'], dataset_name, 'result'))
                    else:
                        metric_data = dict(path_gt=osp.join(self.opt['path']['visualization'], dataset_name, 'driving'), path_generated=osp.join(self.opt['path']['visualization'], dataset_name, 'result'))
                    self.metric_results[metric] = calculate_metric(metric_data, opt_)
                    print(metric, self.metric_results[metric])
                                               
                elif metric in ['pose_accuracy']:
                    metric_data = dict(path_gt=osp.join(self.opt['path']['visualization'], dataset_name, 'driving'), path_generated=osp.join(self.opt['path']['visualization'], dataset_name, 'result'))
                    self.metric_results[metric] = calculate_metric(metric_data, opt_)
                    print(metric, self.metric_results[metric])

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)


    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            print(metric, value)
            log_str += f'\t # {metric}: {value:.4f}\n'
            log_str += f'\t # {metric}: {value}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)


    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['gt'] = self.gt.detach().cpu()
        out_dict['result'] = self.out_dict['out'].detach().cpu()

        out_dict['source'] = self.source.detach().cpu()
        if hasattr(self, 'lq_recon'):
            out_dict['recon'] = self.lq_recon.detach().cpu()

        return out_dict


    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_network(self.motion_estimator, 'net_motion_estimator', current_iter)
        self.save_training_state(epoch, current_iter)

    def make_animation(self, source_img, driving, cpu=False):
        relative = self.opt['val'].get('relative', False)
        adapt_movement_scale = self.opt['val'].get('adapt_scale', False)

        self.gt = driving 
        self.source = source_img.to(self.device)
        self.b = self.gt[0].shape[0]

        predictions = []
        driving_imgs = []

        with torch.no_grad():
            kp_source = self.motion_estimator.estimate_kp(self.source)
            kp_driving_initial = self.motion_estimator.estimate_kp(self.gt[0].to(self.device))
            for frame_idx in tqdm(range(len(self.gt))):
                driving_frame = self.gt[frame_idx].to(self.device)
                kp_driving = self.motion_estimator.estimate_kp(driving_frame)
                kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            
                self.dense_motion = self.motion_estimator.estimate_motion_w_kp(kp_source=kp_source, kp_driving=kp_norm, source_image=self.source)


                logger = get_root_logger()
                w = self.opt['val'].get('w', 1)

                self.out_dict = self.net_g(self.source, self.dense_motion, w=w, inference=True)

                predictions.append(tensor2img([self.out_dict['out'].detach().cpu()], rgb2bgr=True, min_max=(-1, 1)))
                driving_imgs.append(tensor2img([self.gt[frame_idx].detach().cpu()], rgb2bgr=True, min_max=(-1, 1)))

        return predictions, driving_imgs 

    # source: image; driving: video frame list
    def generate_video_image(self, dataloader, current_iter, tb_logger):
        dataset_name = dataloader.dataset.opt['name']

        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}

        driving_dir = osp.join(self.opt['path']['visualization'], dataset_name, 'driving')
        source_dir = osp.join(self.opt['path']['visualization'], dataset_name, 'source')
        result_dir = osp.join(self.opt['path']['visualization'], dataset_name, 'result')

        pbar = tqdm(total=len(dataloader), unit='video')

        self.net_g.eval()
        if self.opt.get('network_motion_estimator', None) is not None:
            self.motion_estimator = build_network(self.opt['network_motion_estimator']).to(self.device)
            self.motion_estimator.eval()
            self.print_network(self.motion_estimator)
            
            # load pretrained models
            load_path = self.opt['path'].get('pretrain_network_motion_estimator', None)
            if load_path is not None:
                param_key = self.opt['path'].get('param_key_m', 'params')
                self.load_network(self.motion_estimator, load_path, self.opt['path'].get('strict_load_motion_estimator', True), param_key)
        
        count_num = 0
        for idx, val_data in enumerate(dataloader):
            video_name = val_data['video_name']
            anchor_idx = val_data['anchor_idx']
            driving_video = val_data['driving_video']
            predictions = []
            visualizations = []

            if val_data['anchor_idx'] is not None:
                driving_forward = driving_video[val_data['anchor_idx']:]
                driving_backward = driving_video[:(val_data['anchor_idx']+1)][::-1]

            predictions_forward, driving_forward_list = self.make_animation(val_data['source'], driving_forward)
            predictions_backward, driving_backward_list = self.make_animation(val_data['source'], driving_backward)
            predictions = predictions_backward[::-1] + predictions_forward[1:]

            drivings = driving_backward_list[::-1] + driving_forward_list[1:]

            source = tensor2img([self.source.detach().cpu()], rgb2bgr=True, min_max=(-1, 1))

            visual = []
            for i in range(len(predictions)):
                vis = np.concatenate((source, drivings[i], predictions[i]), axis=1)
                visual.append(vis)

                img_name = video_name[0] + '_' + val_data['driving_name_list'][i][0]

                imwrite(vis, osp.join(self.opt['path']['visualization'], dataset_name, 'visual', f'{img_name}_v.png'))
                imwrite(predictions[i], osp.join(self.opt['path']['visualization'], dataset_name, 'result', f'{img_name}_r.png'))
                imwrite(source, osp.join(self.opt['path']['visualization'], dataset_name, 'source', f'{img_name}_s.png'))
                imwrite(drivings[i], osp.join(self.opt['path']['visualization'], dataset_name, 'driving', f'{img_name}_d.png'))

                if with_metrics:
                    # calculate metrics
                    for name, opt_ in self.opt['val']['metrics'].items():
                        if name in ['psnr', 'ssim', 'l1']:
                            metric_data = dict(img1=predictions[i], img2=drivings[i])
                            self.metric_results[name] += calculate_metric(metric_data, opt_)
                            count_num += 1

            predictions = [cv2.cvtColor(p, cv2.COLOR_BGR2RGB) for p in predictions]
            visual = [cv2.cvtColor(v, cv2.COLOR_BGR2RGB) for v in visual]
            mimsave(predictions, osp.join(self.opt['path']['visualization'], dataset_name, 'result_videos', f'{video_name[0]}_r.mp4'))
            mimsave(visual, osp.join(self.opt['path']['visualization'], dataset_name, 'visual_videos', f'{video_name[0]}_v.mp4'))
                        
            pbar.update(1)
            pbar.set_description(f'Test {video_name}')
        pbar.close()

        if with_metrics:
            for metric, opt_ in self.opt['val']['metrics'].items():
                if metric in ['psnr', 'ssim', 'l1']:
                    self.metric_results[metric] /= count_num
                    if metric in ['l1']:
                        self.metric_results.update({'l1_255': self.metric_results['l1']/255.})
                elif metric in ['fid']:
                    metric_data = dict(paths=[source_dir, result_dir])

                    self.metric_results[metric] = calculate_metric(metric_data, opt_)
                elif metric in ['lpips']:
                    metric_data = dict(path1=result_dir, path2=driving_dir)

                    self.metric_results[metric] = calculate_metric(metric_data, opt_)
                
                elif metric in ['face_akd']:
                    metric_data = dict(path_gt=driving_dir, path_generated=result_dir)
                    self.metric_results[metric] = calculate_metric(metric_data, opt_)
                
                elif metric in ['face_aed']:
                    if self.opt['val']['cross_id']:
                        metric_data = dict(path_gt=source_dir, path_generated=result_dir)
                    else:
                        metric_data = dict(path_gt=driving_dir, path_generated=result_dir)
                    
                    self.metric_results[metric] = calculate_metric(metric_data, opt_)
                
                elif metric in ['id_similarity']:
                    if self.opt['val']['cross_id']:
                        metric_data = dict(path_gt=source_dir, path_generated=result_dir)
                    else:
                        metric_data = dict(path_gt=driving_dir, path_generated=result_dir)
                    self.metric_results[metric] = calculate_metric(metric_data, opt_)
                    print(metric, self.metric_results[metric])
                                               
                elif metric in ['pose_accuracy']:
                    metric_data = dict(path_gt=driving_dir, path_generated=result_dir)
                    self.metric_results[metric] = calculate_metric(metric_data, opt_)
                    print(metric, self.metric_results[metric])

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)