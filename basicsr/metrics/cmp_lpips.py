import sys

import torch
import os
import lpips
from tqdm import tqdm

from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_lpips(path1,path2):
    with torch.no_grad():
        use_gpu = True         # Whether to use GPU
        spatial = False         # Return a spatial map of perceptual distance.
        # Linearly calibrated models (LPIPS)
        # define as a criterion module (recommended)
        # criterion = lpips.LPIPS(
        #     net_type='alex',  # choose a network type from ['alex', 'squeeze', 'vgg']
        #     version='0.1'  # Currently, v0.1 is supported
        # )
        # loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()

        loss_fn = lpips.LPIPS(net='vgg', spatial=spatial).cuda() # Can also set net = 'squeeze' or 'vgg'
        # loss_fn = lpips.LPIPS(net='alex', spatial=spatial, lpips=False) # Can also set net = 'squeeze' or 'vgg'
        
        ## Example usage with dummy tensors
        im1_path_list = []
        im2_path_list = []
        imgs = os.listdir(path1)
        for im in imgs:
            im1 = os.path.join(path1, im)
            im1_path_list.append(im1)
            im2 = os.path.join(path2, im[:-6]+'_d.png')
            im2_path_list.append(im2)
        dist_ = []
        for i in tqdm(range(len(im1_path_list))):
            dummy_im0 = lpips.im2tensor(lpips.load_image(im1_path_list[i]))
            dummy_im1 = lpips.im2tensor(lpips.load_image(im2_path_list[i]))
            if(use_gpu):
                dummy_im0 = dummy_im0.cuda()
                dummy_im1 = dummy_im1.cuda()
            dist = loss_fn(dummy_im0, dummy_im1)
            dist = loss_fn.forward(dummy_im0, dummy_im1)
            dist_.append(dist.detach().cpu().numpy()[0][0][0][0])
        print('Avarage Distances: %.3f' % (sum(dist_)/len(im1_path_list)))
        return sum(dist_)/len(im1_path_list)