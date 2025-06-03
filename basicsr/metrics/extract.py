import numpy as np
import pandas as pd
import os,sys
from tqdm import tqdm
from skimage.transform import resize
from basicsr.metrics.metric_util import frames2array
from imageio import mimsave
import torch.nn.functional as F
import pdb
from collections import OrderedDict

from basicsr.utils.registry import METRIC_REGISTRY

def extract_face_pose(in_folder, is_video, image_shape, column):
    import face_alignment

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

    out_df = {'file_name': [], 'frame_number': [], 'value': []}

    for file in tqdm(os.listdir(in_folder)):
        video = frames2array(os.path.join(in_folder, file), is_video, image_shape, column)
        for i, frame in enumerate(video):
            kp = fa.get_landmarks(frame)
            if kp is not None:
               kp = kp[0]
            out_df['file_name'].append(file.split('.pn')[0][:-1])
            out_df['frame_number'].append(i)
            out_df['value'].append(kp)

    return pd.DataFrame(out_df)

def extract_face_id(is_video, in_folder, image_shape, column):
    from .OpenFacePytorch.loadOpenFace import prepareOpenFace
    from torch.autograd import Variable
    import torch
    from imageio import mimsave
    net = prepareOpenFace(useCuda=True, gpuDevice=0, useMultiGPU=False).eval()


    out_df = {'file_name': [], 'frame_number': [], 'value': []}

    for file in tqdm(os.listdir(in_folder)):
        video = frames2array(os.path.join(in_folder, file), is_video, image_shape, column)
        for i, frame in enumerate(video):
            frame = frame[..., ::-1]
            frame = resize(frame, (96, 96))
            frame = np.transpose(frame, (2, 0, 1))
            with torch.no_grad():
                frame = Variable(torch.Tensor(frame)).cuda()
                frame = frame.unsqueeze(0)
                id_vec = net(frame)[0].data.cpu().numpy()
            out_df['file_name'].append(file.split('.pn')[0][:-1])
            out_df['frame_number'].append(i)
            out_df['value'].append(id_vec)

    return pd.DataFrame(out_df)

@METRIC_REGISTRY.register()
def cmp_face_akd(path_gt, path_generated, is_video=False, size=[256,256]):

    df1 = extract_face_pose(path_gt, is_video, size, 0)
    df2 = extract_face_pose(path_generated, is_video, size, 0)

    df1 = df1.sort_values(by=['file_name', 'frame_number'])
    df2 = df2.sort_values(by=['file_name', 'frame_number'])

    assert df1.shape == df2.shape

    scores = []
    for i in range(df1.shape[0]):
        try:
            file_name1 = df1['file_name'].iloc[i].split('.')[0]
            file_name2 = df2['file_name'].iloc[i].split('.')[0]
            assert file_name1 == file_name2
            assert df1['frame_number'].iloc[i] == df2['frame_number'].iloc[i]
            if df2['value'].iloc[i] is not None: 
                scores.append(np.mean(np.abs(df1['value'].iloc[i] - df2['value'].iloc[i]).astype(float)))
        except Exception as e:
            print(e)
    print('AKD: ', np.mean(scores))
    return np.mean(scores)

@METRIC_REGISTRY.register()
def cmp_face_aed(path_gt, path_generated, is_video=False, size=[256,256]):

    df1 = extract_face_id(is_video, path_gt, size, 0)
    df2 = extract_face_id(is_video, path_generated, size, 0)

    df1 = df1.sort_values(by=['file_name', 'frame_number'])
    df2 = df2.sort_values(by=['file_name', 'frame_number'])

    assert df1.shape == df2.shape
    scores = []
    for i in range(df1.shape[0]):
        file_name1 = df1['file_name'].iloc[i].split('.')[0]
        file_name2 = df2['file_name'].iloc[i].split('.')[0]
        assert file_name1 == file_name2
        assert df1['frame_number'].iloc[i] == df2['frame_number'].iloc[i]
        scores.append(np.sum(np.abs(df1['value'].iloc[i] - df2['value'].iloc[i]).astype(float) ** 2))
    print('AED: ', np.mean(scores))
    return np.mean(scores)