from insightface.app import FaceAnalysis
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import torch
import sys

from basicsr.utils.registry import METRIC_REGISTRY

def distance(emb1, emb2):
    '''
    cosine similarity
    '''
    return np.dot(emb1, emb2)

def extract_id_from_path(in_folder, size=(256, 256)):
    # Load face detection and recognition package
    app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=size)

    out_df = {'file_name': [], 'value': []}

    for file in tqdm(sorted(os.listdir(in_folder))):
        image_path = os.path.join(in_folder, file)
        img = np.array(Image.open(image_path))[:,:,::-1]
        # Face detection and ID-embedding extraction
        faces = app.get(img)
        if len(faces) == 0:
            print(f"Face detection failed! Please try with another input face image." + file)
            out_df['file_name'].append(file.split('.pn')[0][:-1])
            out_df['value'].append(None)
            continue
        
        faces = faces[0]
        id_emb = faces['embedding']
        id_emb = id_emb/np.linalg.norm(id_emb) 
        out_df['file_name'].append(file.split('.pn')[0][:-1])
        out_df['value'].append(id_emb)
    return pd.DataFrame(out_df)
        
@METRIC_REGISTRY.register()
def calculate_id_similarity(path_gt, path_generated, size=(256,256)):
    df1 = extract_id_from_path(path_gt, size)
    df2 = extract_id_from_path(path_generated, size)

    df1 = df1.sort_values(by=['file_name'])
    df2 = df2.sort_values(by=['file_name'])

    assert df1.shape == df2.shape

    scores = []
    for i in range(df1.shape[0]):
        try:
            file_name1 = df1['file_name'].iloc[i].split('.pn')[0]
            file_name2 = df2['file_name'].iloc[i].split('.pn')[0]
            assert file_name1 == file_name2
            if df2['value'].iloc[i] is not None: 
                scores.append(distance(df1['value'].iloc[i], df2['value'].iloc[i]))
        except Exception as e:
            print(e)
    
    return np.mean(scores)