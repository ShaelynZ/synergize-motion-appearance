import torch
from torch import nn
import torch.nn.functional as F
from basicsr.archs.dense_motion_arch import *
from basicsr.archs.keypoint_detector_arch import *
import pdb

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import collections
from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class Motion_Estimator_keypoint_aware(nn.Module):
    """
    Generator that given source image and and keypoints try to transform image according to movement trajectories
    induced by keypoints. 
    """
    def __init__(self, common_params, dense_motion_params, kp_detector_params):
        super(Motion_Estimator_keypoint_aware, self).__init__()

        if kp_detector_params is not None:
            self.kp_detector = KPDetector(**common_params, **kp_detector_params)
        else:
            raise NotImplementedError('Shoule have kp_detector.')

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(**common_params, **dense_motion_params)
        else:
            raise NotImplementedError('Shoule have dense_motion_network.')
        
    def estimate_kp(self, image): 
        kp = self.kp_detector(image)
        return kp

    def estimate_motion_w_kp(self, kp_source, kp_driving, source_image): 
        dense_motion = self.dense_motion_network(source_image=source_image, kp_driving=kp_driving, kp_source=kp_source)
        dense_motion.update({"kp_driving": kp_driving, "kp_source": kp_source})
        return dense_motion
    
    def forward(self, driving_image, source_image, only_return_kp_driving=False, relative=False):
        kp_driving = self.kp_detector(driving_image)
        if only_return_kp_driving:
            return kp_driving
        kp_source = self.kp_detector(source_image, isSource=True)

        dense_motion = self.dense_motion_network(source_image=source_image, kp_driving=kp_driving, kp_source=kp_source)
        
        dense_motion.update({"kp_driving": kp_driving, "kp_source": kp_source})

        return dense_motion