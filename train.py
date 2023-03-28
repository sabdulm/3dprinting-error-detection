
import os
import cv2

import pandas as pd

import numpy as np
import torch


from CustomDataset import CustomTensorDataset
from torch.utils.data import DataLoader


from transformers import AutoFeatureExtractor, AutoModelForImageClassification


import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from lightning_model import LightningModel

from utils import get_images_and_targets

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


np.random.seed(42)

ROOT_DATA_PATH = '/tmp/'
SAVING_OUTPUTS = './'
train_set = ROOT_DATA_PATH + 'train.csv'
test_set = ROOT_DATA_PATH + 'test.csv'
images = ROOT_DATA_PATH + 'images/'
MODEL_NAME = 'microsoft/resnet-50'


# HYPERPARAMETERS
LEARNING_RATE = 0.0000001
EPOCHS = 100
BATCH_SIZE = 100



def main():

    print("loading train and test data")

    train_labels = pd.read_csv(train_set)

    image_processor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)

    

    trainX, trainY = get_images_and_targets(train_labels, images, image_processor, train_fraction=0.8)
    valX, valY = get_images_and_targets(train_labels, images, image_processor, test=False, val=True, train_fraction=0.8)

    print("loaded train and val data")

    dataset_train = CustomTensorDataset(trainX, trainY)
    dataset_val = CustomTensorDataset(valX, valY)
    

    trainloader = DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
    valloader = DataLoader(dataset=dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=6)    

    print("data ready for training")
  
    pretrained_model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={str(i): c for i, c in enumerate(range(2))},
        label2id={c: str(i) for i, c in enumerate(range(2))},
        ignore_mismatched_sizes = True,
        # hidden_dropout_prob=0.3,
        # attention_probs_dropout_prob=0.3
    )




    light_model = LightningModel(pretrained_model, num_classes=2,reshape_input=False, learning_rate=LEARNING_RATE)

    print("model loaded")

    if not os.path.exists('{}{}/'.format(SAVING_OUTPUTS, 'output/model')):
        os.makedirs('{}{}/'.format(SAVING_OUTPUTS, 'output/model'))
    

    checkpoint_callback = ModelCheckpoint(
        monitor='val_f1',
        dirpath='{}{}/'.format(SAVING_OUTPUTS, 'output/model'),
        filename='{}-{}-{}-{}'.format(MODEL_NAME, '3dprint', LEARNING_RATE, BATCH_SIZE)+'-{epoch:02d}-{val_f1:.4f}_cv_updated',
        save_top_k=3,
        mode='max',
    )
    early_stopping = EarlyStopping(monitor="val_f1", min_delta=0.00, patience=10, verbose=False, mode="max")


    logger = TensorBoardLogger('lightning_logs', name=f'{MODEL_NAME}_lr_{LEARNING_RATE}_cv_updated_epoch_{EPOCHS}')


    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        precision=16,
        accelerator='gpu', devices=[0],
        num_sanity_val_steps=0,
        # check_val_every_n_epoch=5,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        # strategy='ddp'
    )

    print('Start Training...')
    trainer.fit(light_model, trainloader, valloader)


    print("training done")



if __name__ == '__main__':
    main()