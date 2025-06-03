import os
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from imageio import mimread
import imageio
import torchvision.transforms as T
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from basicsr.data.augmentation import AllAugmentationTransform
import glob
from PIL import Image
import pdb
import glob

import cv2

import decord
from decord import VideoReader, cpu

from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor, tensor2img, imwrite
from basicsr.data.data_util import paths_from_folder
from basicsr.data.transforms import augment, augment_video
from basicsr.utils.registry import DATASET_REGISTRY
from torchvision.transforms.functional import (normalize)

def read_video(name, frame_shape):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """

    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = np.array(
            [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)])
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + frame_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video = np.array(mimread(name,memtest=False))
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array

@DATASET_REGISTRY.register()
class FramesMotionTransferDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - folder with all frames
    
    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            root_dir (str): Data root path for gt.
            io_backend (dict): IO backend type and other kwarg.
            mean (list | tuple): Image mean.
            std (list | tuple): Image std.
            use_hflip (bool): Whether to horizontally flip.
    """

    def __init__(self, opt):
        super(FramesMotionTransferDataset, self).__init__()
        logger = get_root_logger()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.root_dir = opt['root_dir']
        
        self.gt_size = opt.get('gt_size', 512)
        
        self.mean = opt.get('mean', [0.5, 0.5, 0.5])
        self.std = opt.get('std', [0.5, 0.5, 0.5])

        self.videos = os.listdir(self.root_dir)
        
        self.id_sampling = opt.get('id_sampling', False)

        self.is_train = opt.get('is_train', True)

        if os.path.exists(os.path.join(self.root_dir, 'train')):
            assert os.path.exists(os.path.join(self.root_dir, 'test'))
            logger.info("Use predefined train-test split.")
            if self.id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(self.root_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(self.root_dir, 'train'))
            test_videos = os.listdir(os.path.join(self.root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if self.is_train else 'test')
        else:
            raise NotImplementedError("random train-test split not implemented!")

        if self.is_train:
            self.videos = train_videos
        else:
            self.videos = test_videos

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        if self.is_train and self.id_sampling:
            name = self.videos[idx]
            path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
        else:
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)

        video_name = os.path.basename(path)

        if self.is_train and os.path.isdir(path):
            frames = sorted(glob.glob(path+'/*.png'))
            num_frames = len(frames)
            if num_frames == 0:
                frames = sorted(glob.glob(path+'/*.jpg'))
                num_frames = len(frames)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))

            video_array = []

            for idx in frame_idx:
                try:
                    img_bytes = self.file_client.get(frames[idx])
                    video_array.append(imfrombytes(img_bytes, float32=True))
                except Exception as e:
                    print(e)
        else:
            video_array = read_video(path, frame_shape=self.frame_shape)
            num_frames = len(video_array)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) if self.is_train else range(
                num_frames)
            video_array = video_array[frame_idx]
        
        video_array, status = augment_video(video_array, hflip=self.opt['use_hflip'], rotation=False, time_flip=self.opt['use_time_flip'], return_status=True)

        if video_array[0].shape[-2] != self.gt_size:
            video_array[0] = cv2.resize(video_array[0], (int(self.gt_size), int(self.gt_size)), interpolation=cv2.INTER_LINEAR)
            video_array[1] = cv2.resize(video_array[1], (int(self.gt_size), int(self.gt_size)), interpolation=cv2.INTER_LINEAR)

        # BGR to RGB, HWC to CHW, numpy to tensor
        source, driving = img2tensor([video_array[0], video_array[1]], bgr2rgb=True, float32=True)
        # Set vgg range_norm=True if use the normalization here
        # normalize
        normalize(source, self.mean, self.std, inplace=True)
        normalize(driving, self.mean, self.std, inplace=True)

        return_dict = {'source': source, 'driving': driving}
        return return_dict

@DATASET_REGISTRY.register()
class FramesMotionTransferTestDataset_CrossID_videopair_anchor(Dataset):
    """
    Dataset for evaluation. pairs_list contains: source (frame), driving (video), anchor (frame), anchor_idx

    Dataset of videos, each video can be represented as:
      - folder with all frames
    
    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            root_dir (str): Data root path for gt.
            io_backend (dict): IO backend type and other kwarg.
            mean (list | tuple): Image mean.
            std (list | tuple): Image std.
            use_hflip (bool): Whether to horizontally flip.
    """
    def __init__(self, opt):
        super(FramesMotionTransferTestDataset_CrossID_videopair_anchor, self).__init__()

        self.opt = opt
        self.root_dir = opt['root_dir']

        self.gt_size = opt.get('gt_size', 512)

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        logger = get_root_logger()
        logger.info(f'Generate data info for VideoTestDataset - {opt["name"]}')

        self.mean = opt.get('mean', [0.5, 0.5, 0.5])
        self.std = opt.get('std', [0.5, 0.5, 0.5])

        self.max_frame = opt.get('max_frame', None)

        pairs_list = opt.get('pairs_list', None)

        if pairs_list is not None:
            pairs = pd.read_csv(pairs_list)

            self.source = pairs['source'].tolist()
            self.driving = pairs['driving'].tolist() # videos
            if 'anchor' in pairs:
                self.anchors = pairs['anchor'].tolist()
            else:
                self.anchors = None

            if 'anchor_idx' in pairs:
                self.anchor_idx = pairs['anchor_idx'].tolist()
            else:
                self.anchor_idx = None
        else:
            raise NotImplementedError(f'Shoule provide cross id pairs for dataset.')

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        path_source = self.source[idx]
        path_driving = self.driving[idx]
        if self.anchors is not None:
            path_anchor = self.anchors[idx]
        else:
            path_anchor = None
        if self.anchor_idx is not None:
            anchor_idx = self.anchor_idx[idx]
        else:
            anchor_idx = None

        video_name = os.path.basename(os.path.dirname(path_source))[:-4]+'_'+ os.path.basename(path_source)[:-4]+'_'+os.path.basename(path_driving)[:-4]

        driving = []
        driving_name = []

        img_bytes = self.file_client.get(path_source)
        source = imfrombytes(img_bytes, float32=True)

        if os.path.isdir(path_driving): 
            frames = sorted(glob.glob(path_driving+'/*.png'))
            num_frames = len(frames)
            if num_frames == 0:
                frames = sorted(glob.glob(path_driving+'/*.jpg'))
                num_frames = len(frames)
        
                    
            if self.max_frame is not None:
                if num_frames -1 > self.max_frame:
                    num_frames = self.max_frame + 1

            for i in range(num_frames-1): 
                img_bytes = self.file_client.get(frames[i+1])
                driving.append(imfrombytes(img_bytes, float32=True))
                driving_name.append(os.path.basename(frames[i+1]))

        if path_anchor is not None:
            img_bytes = self.file_client.get(path_anchor)
            anchor = imfrombytes(img_bytes, float32=True)
        else:
            print('path_anchor is None, using driving[0]')
            anchor = driving[0]
            anchor_idx = 0

        if source.shape[-2] != self.gt_size:
            source = cv2.resize(source, (int(self.gt_size), int(self.gt_size)), interpolation=cv2.INTER_LINEAR)
            driving = [cv2.resize(frame, (int(self.gt_size), int(self.gt_size)), interpolation=cv2.INTER_LINEAR) for frame in driving]
            anchor = cv2.resize(anchor, (int(self.gt_size), int(self.gt_size)), interpolation=cv2.INTER_LINEAR)

        # BGR to RGB, HWC to CHW, numpy to tensor
        if anchor is not None:
            source, anchor = img2tensor([source, anchor], bgr2rgb=True, float32=True)
        else:
            source = img2tensor(source, bgr2rgb=True, float32=True)
        driving = [img2tensor(frame, bgr2rgb=True, float32=True) for frame in driving]

        # Set vgg range_norm=True if use the normalization here
        # normalize
        normalize(source, self.mean, self.std, inplace=True)
        if anchor is not None:
            normalize(anchor, self.mean, self.std, inplace=True)
        for i in range(len(driving)):
            normalize(driving[i], self.mean, self.std, inplace=True)

        return_dict = {'source': source, 'driving_video': driving, 'anchor': anchor, 'video_name': video_name, 'driving_name_list': driving_name, 'anchor_idx': anchor_idx}

        return return_dict

@DATASET_REGISTRY.register()
class FramesMotionTransferTestDataset_PairsList(Dataset):
    """
    Dataset for evaluation. pairs_list contains: source (frame), driving (frame), anchor (frame)

    Dataset of videos, each video can be represented as:
      - folder with all frames
    
    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            root_dir (str): Data root path for gt.
            io_backend (dict): IO backend type and other kwarg.
            mean (list | tuple): Image mean.
            std (list | tuple): Image std.
            use_hflip (bool): Whether to horizontally flip.
    """
    def __init__(self, opt):
        super(FramesMotionTransferTestDataset_PairsList, self).__init__()

        self.opt = opt

        self.root_dir = opt['root_dir']

        self.gt_size = opt.get('gt_size', 512)

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        logger = get_root_logger()
        logger.info(f'Generate data info for VideoTestDataset - {opt["name"]}')

        self.mean = opt.get('mean', [0.5, 0.5, 0.5])
        self.std = opt.get('std', [0.5, 0.5, 0.5])

        self.videos = os.listdir(self.root_dir)

        pairs_list = opt.get('pairs_list', None)

        if pairs_list is not None:
            pairs = pd.read_csv(pairs_list)

            self.source = pairs['source'].tolist()
            self.driving = pairs['driving'].tolist()
            if 'anchor' in pairs:
                self.anchors = pairs['anchor'].tolist()
            else:
                self.anchors =self.driving
        else:
            raise NotImplementedError(f'Shoule provide pairs_list for dataset.')

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        path_source = self.source[idx]
        path_driving = self.driving[idx]
        path_anchor = self.anchors[idx]
        frame_name = os.path.basename(os.path.dirname(path_source))[:-4]+'_'+ os.path.basename(path_source)[:-4]+'_'+os.path.basename(os.path.dirname(path_driving))[:-4]+'_'+ os.path.basename(path_driving)[:-4]

        try:
            img_bytes = self.file_client.get(path_source)
            source = imfrombytes(img_bytes, float32=True)
            if source.shape[-2]!=self.gt_size:
                source = cv2.resize(source, (int(self.gt_size), int(self.gt_size)), interpolation=cv2.INTER_LINEAR)

            img_bytes = self.file_client.get(path_driving)
            driving = imfrombytes(img_bytes, float32=True)
            if driving.shape[-2]!=self.gt_size:
                driving = cv2.resize(driving, (int(self.gt_size), int(self.gt_size)), interpolation=cv2.INTER_LINEAR)

            img_bytes = self.file_client.get(path_anchor)
            anchor = imfrombytes(img_bytes, float32=True)
            if anchor.shape[-2]!=self.gt_size:
                anchor = cv2.resize(anchor, (int(self.gt_size), int(self.gt_size)), interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            print(e)

        # BGR to RGB, HWC to CHW, numpy to tensor
        source, driving, anchor = img2tensor([source, driving, anchor], bgr2rgb=True, float32=True)

        # Set vgg range_norm=True if use the normalization here
        # normalize
        normalize(source, self.mean, self.std, inplace=True)
        normalize(driving, self.mean, self.std, inplace=True)
        normalize(anchor, self.mean, self.std, inplace=True)

        return_dict = {'source': source, 'driving': driving, 'anchor': anchor, 'frame_name': frame_name}
        return return_dict
