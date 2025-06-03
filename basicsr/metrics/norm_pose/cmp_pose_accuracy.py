from .utils.mp_utils_refine  import LMKExtractor
import cv2
#from .utils.pose_util import smooth_pose_seq,project_points,matrix_to_euler_and_translation, euler_and_translation_to_matrix, project_points_with_trans,invert_projection
from .utils.pose_util import matrix_to_euler_and_translation
import pdb

from tqdm import tqdm
import os
import pandas as pd
import numpy as np

from basicsr.utils.registry import METRIC_REGISTRY

def distance(emb1, emb2):
    '''
    L1 distance
    '''
    return np.mean(np.absolute(emb1 - emb2))

def extract_pose_from_path(in_folder):
    ref_lmk_extractor = LMKExtractor()

    out_df = {'file_name': [], 'value': []}

    for file in tqdm(sorted(os.listdir(in_folder))):
        image_path = os.path.join(in_folder, file)

        ref_result = ref_lmk_extractor(cv2.imread(image_path))
        try:
            euler_angles, translation_vector = matrix_to_euler_and_translation(ref_result['trans_mat'])

            out_df['file_name'].append(file.split('.pn')[0][:-1])
            out_df['value'].append(euler_angles)
        except Exception as e:
            print(e)
            out_df['file_name'].append(file.split('.pn')[0][:-1])
            out_df['value'].append(None)
    return pd.DataFrame(out_df)

@METRIC_REGISTRY.register()
def calculate_pose_distance(path_gt, path_generated):
    df1 = extract_pose_from_path(path_gt)
    df2 = extract_pose_from_path(path_generated)

    df1 = df1.sort_values(by=['file_name'])
    df2 = df2.sort_values(by=['file_name'])

    assert df1.shape == df2.shape

    dis = []
    for i in range(df1.shape[0]):
        try:
            file_name1 = df1['file_name'].iloc[i].split('.pn')[0]
            file_name2 = df2['file_name'].iloc[i].split('.pn')[0]
            assert file_name1 == file_name2
            if df2['value'].iloc[i] is not None: 
                dis.append(distance(df1['value'].iloc[i], df2['value'].iloc[i]))
        except Exception as e:
            print(e)
    
    return np.mean(dis)