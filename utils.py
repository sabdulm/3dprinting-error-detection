import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoFeatureExtractor


# PRINT_IDS_FOR_VAL_ONLY = [1672777638, 1672773342,1672767789, 1679265796, 1678415191]
PRINT_IDS_FOR_VAL_ONLY = [1678816535, 1678736713]
PRINTERS_FOR_VAL_ONLY = []


def get_images_and_targets(labels_df: pd.DataFrame, images_path: str, image_processor: AutoFeatureExtractor, test=False, train_fraction=0.7, val=False):
    
    
    raw_labels = labels_df.values
    if test == False:
        filter = (labels_df['printer_id'].isin(PRINTERS_FOR_VAL_ONLY) | labels_df['print_id'].isin(PRINT_IDS_FOR_VAL_ONLY))

        if val == True:
            raw_labels = labels_df[filter].values
        else:
            raw_labels = labels_df[~filter].values
    else:
        raw_labels = labels_df.values

    # if test==False:
    #     fraction = int(len(raw_labels) - (len(raw_labels)*train_fraction))
    #     r_indexes = np.arange(len(raw_labels))
    #     np.random.shuffle(r_indexes)
    #     raw_labels = raw_labels[r_indexes]
    #     if val:
    #         raw_labels = raw_labels[:fraction , :]
    #     else:
    #         raw_labels = raw_labels[fraction: , :]

    
    
    X, Y = [], []
    # print(len(raw_labels)//10000)
    for i in tqdm(range(len(raw_labels))):
        image = Image.open(images_path + raw_labels[i][0])
        image = image_processor(image, return_tensors='pt')

        X.append(image.pixel_values)
        if test==False:
            Y.append(raw_labels[i][3])
        else:
            Y.append(raw_labels[i][0])
    # print(X)

    X, Y = torch.vstack(X), torch.from_numpy(np.array(Y)).reshape(-1,1) if test == False else np.array(Y)
    return X, Y