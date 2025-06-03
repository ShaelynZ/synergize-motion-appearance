from torch import nn
import torch.nn.functional as F
import torch
from basicsr.utils.motion_estimator_util import Hourglass, AntiAliasInterpolation2d, make_coordinate_grid, kp2gaussian
import pdb
from basicsr.utils.motion_estimator_util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
import math

from basicsr.utils.registry import ARCH_REGISTRY
import collections

@ARCH_REGISTRY.register()
class DenseMotionNetwork(nn.Module):
    """
    Module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
    """

    def __init__(self, block_expansion, num_blocks, max_features, num_kp, num_channels, estimate_occlusion_map=False,
                 scale_factor=1, kp_variance=0.01, multi_mask=False, occlusion_num=5, model_path=None):
        super(DenseMotionNetwork, self).__init__()
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_kp + 1) * (num_channels + 1),
                                   max_features=max_features, num_blocks=num_blocks)

        self.mask = nn.Conv2d(self.hourglass.out_filters, num_kp + 1, kernel_size=(7, 7), padding=(3, 3))

        self.num_kp = num_kp
        self.scale_factor = scale_factor
        self.kp_variance = kp_variance
        self.multi_mask = multi_mask

        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)
        
        if estimate_occlusion_map:
            if multi_mask:
                up = []
                self.up_nums = int(math.log(1/scale_factor, 2))
                self.occlusion_num = occlusion_num

                hourglass_output_size = self.hourglass.out_channels
                
                channel = [hourglass_output_size[-1]//(2**i) for i in range(self.up_nums)]
                for i in range(self.up_nums):
                    up.append(UpBlock2d(channel[i], channel[i]//2, kernel_size=3, padding=1))
                self.up = nn.ModuleList(up)

                channel = [hourglass_output_size[-i-1] for i in range(self.occlusion_num-self.up_nums)[::-1]]
                for i in range(self.up_nums):
                    channel.append(hourglass_output_size[-1]//(2**(i+1)))
                occlusion = []
                
                for i in range(self.occlusion_num):
                    occlusion.append(nn.Conv2d(channel[i], 1, kernel_size=(7, 7), padding=(3, 3)))
                self.occlusion = nn.ModuleList(occlusion)
            else:
                self.occlusion = nn.Conv2d(self.hourglass.out_filters, 1, kernel_size=(7, 7), padding=(3, 3))
        else:
            self.occlusion = None

        if model_path is not None:
            checkpoint = torch.load(model_path, map_location='cpu')
            ckp_densemotion = collections.OrderedDict((k.replace('module.dense_motion_network.',''),v) for k,v in checkpoint['generator'].items() if k.startswith('module.dense_motion_network.'))
            self.load_state_dict(ckp_densemotion)

    def create_heatmap_representations(self, source_image, kp_driving, kp_source):
        """
        Eq 6. in the paper H_k(z)
        """
        spatial_size = source_image.shape[2:]
        gaussian_driving = kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=self.kp_variance)
        gaussian_source = kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=self.kp_variance)
        heatmap = gaussian_driving - gaussian_source
        #adding background feature
        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1]).type(heatmap.type())
        heatmap = torch.cat([zeros, heatmap], dim=1)
        heatmap = heatmap.unsqueeze(2)
        return heatmap
    
    def create_driving_heatmap_representations(self, source_image, kp_driving):
        spatial_size = source_image.shape[2:]
        gaussian_driving = kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=self.kp_variance)
        return gaussian_driving

    def create_sparse_motions(self, source_image, kp_driving, kp_source):
        """
        Eq 4. in the paper T_{s<-d}(z)
        """
        bs, _, h, w = source_image.shape
        identity_grid = make_coordinate_grid((h, w), type=kp_source['value'].type())
        identity_grid = identity_grid.view(1, 1, h, w, 2)
        coordinate_grid = identity_grid - kp_driving['value'].view(bs, self.num_kp, 1, 1, 2)
        if 'jacobian' in kp_driving and kp_driving['jacobian'] is not None:
            jacobian = torch.matmul(kp_source['jacobian'], torch.inverse(kp_driving['jacobian']))
            jacobian = jacobian.unsqueeze(-3).unsqueeze(-3)
            jacobian = jacobian.repeat(1, 1, h, w, 1, 1)
            coordinate_grid = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))
            coordinate_grid = coordinate_grid.squeeze(-1)

        driving_to_source = coordinate_grid + kp_source['value'].view(bs, self.num_kp, 1, 1, 2)

        #adding background feature
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1) 
        sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1) #bs, num_kp+1,w,h,2
        return sparse_motions

    def create_deformed_source_image(self, source_image, sparse_motions):
        """
        Eq 7. in the paper \hat{T}_{s<-d}(z)
        """
        bs, _, h, w = source_image.shape
        source_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, self.num_kp + 1, 1, 1, 1, 1)
        source_repeat = source_repeat.view(bs * (self.num_kp + 1), -1, h, w)
        sparse_motions = sparse_motions.view((bs * (self.num_kp + 1), h, w, -1))
        sparse_deformed = F.grid_sample(source_repeat, sparse_motions)
        sparse_deformed = sparse_deformed.view((bs, self.num_kp + 1, -1, h, w))
        return sparse_deformed

    def forward(self, source_image, kp_driving, kp_source):
        if self.scale_factor != 1:
            source_image = self.down(source_image)
        bs, _, h, w = source_image.shape
        out_dict = dict()
        heatmap_representation = self.create_heatmap_representations(source_image, kp_driving, kp_source)
        sparse_motion = self.create_sparse_motions(source_image, kp_driving, kp_source)
        deformed_source = self.create_deformed_source_image(source_image, sparse_motion)
        out_dict['sparse_motion'] = sparse_motion
        out_dict['sparse_deformed'] = deformed_source

        input = torch.cat([heatmap_representation, deformed_source], dim=2)
        input = input.view(bs, -1, h, w)

        prediction = self.hourglass(input, mode = 1)

        mask = self.mask(prediction[-1])
        mask = F.softmax(mask, dim=1)
        out_dict['mask'] = mask
        mask = mask.unsqueeze(2)
        sparse_motion = sparse_motion.permute(0, 1, 4, 2, 3)
        deformation = (sparse_motion * mask).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 1)

        out_dict['deformation'] = deformation
        out_dict['kp_heatmap'] = heatmap_representation.squeeze(2) # b, num_kp+1, (1), h, w
        out_dict['driving_kp_heatmap'] = self.create_driving_heatmap_representations(source_image, kp_driving) # b, num_kp+1, h, w
        out_dict['source'] = source_image

        # Sec. 3.2 in the paper
        if self.occlusion:
            if self.multi_mask:
                occlusion_map = []
                for i in range(self.occlusion_num-self.up_nums):
                    occlusion_map.append(torch.sigmoid(self.occlusion[i](prediction[self.up_nums-self.occlusion_num+i])))
                prediction = prediction[-1]
                for i in range(self.up_nums):
                    prediction = self.up[i](prediction)
                    occlusion_map.append(torch.sigmoid(self.occlusion[i+self.occlusion_num-self.up_nums](prediction)))
            else:
                occlusion_map = torch.sigmoid(self.occlusion(prediction[-1]))
            out_dict['occlusion_map'] = occlusion_map

        return out_dict