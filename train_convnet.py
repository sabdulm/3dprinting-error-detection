
import os
# from convnet import ConvNet
import cv2

import pandas as pd

import numpy as np
import torch


from CustomDataset import CustomTensorDataset
from torch.utils.data import DataLoader


from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from caxton_model.network_module import ParametersClassifier


import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics
import torch.nn.functional as F

from lightning_model import LightningModel

from utils import get_images_and_targets

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


# np.random.seed(42)

ROOT_DATA_PATH = '/s/babbage/e/nobackup/nkrishna/m3x/others/data/'
SAVING_OUTPUTS = './'
train_set = ROOT_DATA_PATH + 'train.csv'
test_set = ROOT_DATA_PATH + 'test.csv'
images = ROOT_DATA_PATH + 'images/'
MODEL_NAME = 'microsoft/swin-tiny-patch4-window7-224'


# HYPERPARAMETERS
LEARNING_RATE = 1e-03
EPOCHS = 10
BATCH_SIZE = 100

# import numpy as np
# import torch
# import torchmetrics
# import torch.nn.functional as F

        
class ConvNet(pl.LightningModule):
    
    def __init__(self, n_outputs, learning_rate=0.01, optimizer='adam'):
        
        super().__init__()

        self.n_class = n_outputs
        self.lr = learning_rate

        self.opt = optimizer
        self.criterion = torch.nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax(dim=1)

        self.batch_norm1 = torch.nn.BatchNorm2d(64)
        # self.batch_norm2 = torch.nn.BatchNorm2d(192)
        # self.batch_norm3 = torch.nn.BatchNorm2d(384)
        # self.batch_norm4 = torch.nn.BatchNorm2d(256)
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = torch.nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = torch.nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((6, 6))
        self.dropout1 = torch.nn.Dropout(0.3)
        self.fc1 = torch.nn.Linear(256 * 6 * 6, 4096)
        self.dropout2 = torch.nn.Dropout(0.3)
        self.fc2 = torch.nn.Linear(4096, 4096)
        self.dropout3 = torch.nn.Dropout(0.3)
        self.fc3 = torch.nn.Linear(4096, self.n_class)
        
        # self.to(self.device)


    def forward(self, X):
        # Ys = self.forward_all_outputs(X)
        # return Ys[-1]

        x = self.relu(self.conv1(X))
        x = self.maxpool(x)
        x = self.batch_norm1(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        # x = self.batch_norm2(x)
        x = self.relu(self.conv3(x))
        # x = self.batch_norm3(x)
        x = self.relu(self.conv4(x))
        # x = self.batch_norm4(x)
        x = self.relu(self.conv5(x))
        # x = self.batch_norm()
        x = self.maxpool(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.dropout1(x)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.relu(self.fc2(x))
        x = self.dropout3(x)
        return self.fc3(x)
    
    def training_step(self, batch, batch_idx):
        img_seq, label = batch
        gt_label = F.one_hot(label.view(1, -1)[0].long(), num_classes=self.n_class).float()
        

        logits = self.forward(img_seq) # output size: batch_size x 4
        probs = self.softmax(logits)
        preds = torch.max(probs, 1, keepdim=True)[1].int()
        
        loss = self.criterion(logits, gt_label)

        f1 = torchmetrics.functional.f1_score(preds, label, task='binary', average='weighted')
        self.log('train_f1_batch', f1, prog_bar=True)

        return {'loss': loss, 'f1': f1}

    def training_epoch_end(self, outputs):
        # log epoch metric

        loss = sum(output['loss'] for output in outputs) / len(outputs)
        self.logger.experiment.add_scalar("Loss/Train", loss, self.current_epoch)
        self.log('train_loss', loss, prog_bar=True)

        top1 = sum(output['f1'] for output in outputs) / len(outputs)
        self.logger.experiment.add_scalar("F1/Train", top1, self.current_epoch)
        self.log('train_f1', top1, prog_bar=True)

        

    def validation_step(self, batch, batch_idx):
        img_seq, label = batch
        gt_label = F.one_hot(label.view(1, -1)[0].long(), num_classes=self.n_class).float()

        
        logits = self.forward(img_seq) # output size: batch_size x 49
        probs = self.softmax(logits)
        preds = torch.max(probs, 1, keepdim=True)[1].int()

        loss = self.criterion(logits, gt_label)
        
        f1 = torchmetrics.functional.f1_score(preds, label, task='binary', average='weighted')
        return {'loss':loss, 'f1': f1}



    
    def validation_epoch_end(self, outputs):

        loss = sum(output['loss'] for output in outputs) / len(outputs)
        self.logger.experiment.add_scalar("Loss/Validation", loss, self.current_epoch)
        self.log('val_loss', loss, prog_bar=True)

        top1 = sum(output['f1'] for output in outputs) / len(outputs)
        self.logger.experiment.add_scalar("F1/Validation", top1, self.current_epoch)
        self.log('val_f1', top1, prog_bar=True)

    

    def predict_step(self, batch, batch_idx):
        img_seq, label = batch
        logits = self.forward(img_seq)
        probs = self.softmax(logits)

        preds = torch.max(probs, 1, keepdim=True)[1].int()
        return probs, preds

    def configure_optimizers(self):
        
        return torch.optim.SGD(self.parameters(), lr=self.lr) if self.opt =='sgd' else torch.optim.Adam(self.parameters(), lr=self.lr)
    


    def use(self, X):
        # Set input matrix to torch.tensors if not already.
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).float()
        
        X = torch.permute(X, (0, 3, 1, 2))
        Y = self.forward(X)
        probs = self.softmax(Y)
        classes = self.classes[torch.argmax(probs, axis=1).cpu().numpy()]
        return classes.cpu().numpy(), probs.detach().cpu().numpy()



def main():

    print("loading train and test data")

    train_labels = pd.read_csv(train_set)

    image_processor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)

    

    trainX, trainY = get_images_and_targets(train_labels, images, image_processor)
    # valX, valY = get_images_and_targets(train_labels, images, image_processor, test=False, val=True, train_fraction=0.8)

    print("loaded train and val data")

    dataset_train = CustomTensorDataset(trainX, trainY)
    # dataset_val = CustomTensorDataset(valX, valY)
    

    trainloader = DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
    # valloader = DataLoader(dataset=dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=6)    

    print("data ready for training")
  

    model = ConvNet(2,learning_rate=LEARNING_RATE)



    # light_model = LightningModel(pretrained_model, num_classes=2,reshape_input=False, learning_rate=LEARNING_RATE)

    print("model loaded")

    if not os.path.exists('{}{}/'.format(SAVING_OUTPUTS, 'output/alex_net')):
        os.makedirs('{}{}/'.format(SAVING_OUTPUTS, 'output/alex_net'))
    

    checkpoint_callback = ModelCheckpoint(
        monitor='train_f1',
        dirpath='{}{}/'.format(SAVING_OUTPUTS, 'output/alex_net'),
        filename='{}-{}-{}-{}'.format(MODEL_NAME, '3dprint_convnet', LEARNING_RATE, BATCH_SIZE)+'-{epoch:02d}-{train_f1:.4f}_with_dropout_03_batchnorm_first_conv',
        save_top_k=10,
        mode='max',
    )
    early_stopping = EarlyStopping(monitor="train_f1", min_delta=0.00, patience=10, verbose=False, mode="max")


    logger = TensorBoardLogger('lightning_logs', name=f'{MODEL_NAME}_convnet_lr_{LEARNING_RATE}_with_dropout_03_batchnorm_first_conv_epoch_{EPOCHS}')


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
    trainer.fit(model, trainloader)


    print("training done")

    



if __name__ == '__main__':
    main()