import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import classification_report
from CustomDataset import CustomTensorDataset
from torch.utils.data import DataLoader
from utils import get_images_and_targets
# from caxton_model.network_module import ParametersClassifier
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics
from transformers import AutoFeatureExtractor, AutoModelForImageClassification


from lightning_model import LightningModel

# from convnet import ConvNet

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

CHECKPOINT_MODEL_PATH = './output/alex_net/microsoft/swin-tiny-patch4-window7-224-3dprint_convnet-0.001-100-epoch=09-train_f1=0.9950_with_dropout_03_batchnorm_first_conv.ckpt'

        
        
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
    # train_labels = pd.read_csv(train_set)
    test_labels = pd.read_csv(test_set)


    image_processor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)

    # valX, valY = get_images_and_targets(train_labels, images, image_processor, test=False, val=True, train_fraction=0.8)
    # dataset_val = CustomTensorDataset(valX, valY)
    # valloader = DataLoader(dataset=dataset_val, batch_size=100, shuffle=False, num_workers=6)

    testX, imgpathsY = get_images_and_targets(test_labels, images, image_processor, test=True)
    dataset_test = CustomTensorDataset(testX, torch.ones(len(imgpathsY)).reshape(-1,1))
    testloader = DataLoader(dataset=dataset_test, batch_size=100, shuffle=False, num_workers=6)


    # cax_model = ParametersClassifier(2).load_from_checkpoint(CHECKPOINT_MODEL_PATH, num_classes=2).eval()
    model = ConvNet(2, learning_rate=LEARNING_RATE).load_from_checkpoint(CHECKPOINT_MODEL_PATH, n_outputs=2, learning_rate=LEARNING_RATE)
    
    device = torch.device('cuda:1')
    model = model.cuda(device).eval()
    # cax_model = cax_model.to(device)

    softmax = torch.nn.Softmax(dim=1)

    # for evaluating performance on validation set
    # gt_max, pred_max, probs_all = [], [], []
    # for loader in [valloader]:
    #     with torch.no_grad():

    #         for idx, data in tqdm(enumerate(loader)):

    #             img_seq, label = data
    #             img_seq = img_seq.cuda(device)
    #             logits = model(img_seq)


    #             probs = softmax(logits)
    #             preds = torch.max(probs, 1, keepdim=True)[1].int().cpu()
                
                
    #             gt_max.append(label)
    #             pred_max.append( preds)
    #             probs_all.append(probs)


    #     gt_max = torch.vstack(gt_max).cpu()
    #     pred_max = torch.vstack(pred_max).cpu()
    #     probs_all = torch.vstack(probs_all).cpu()

    #     print(classification_report(gt_max, pred_max, zero_division=0, digits=7))


    # get predictions for test set 
    gt_max, pred_max, probs_all = [], [], []
    with torch.no_grad():

        for idx, data in tqdm(enumerate(testloader)):

            img_seq, label = data
            
            img_seq = img_seq.cuda(device)
            
            logits = model(img_seq)


            probs = softmax(logits)
            preds = torch.max(probs, 1, keepdim=True)[1].int().cpu()
            
            
            pred_max.append( preds)


    test_predictions = torch.vstack(pred_max).cpu()
    

    result = np.hstack((imgpathsY.reshape(-1,1), test_predictions.numpy()))

    results_df = pd.DataFrame(result, columns=['img_path', 'has_under_extrusion'])
    saving_name=MODEL_NAME.replace('/','_')
    results_df.to_csv(SAVING_OUTPUTS+f'results_{saving_name}_alex_net_with_dropout_03_epoch_09_with_first_batch_norm.csv', index=False)


if __name__ =='__main__':
    main()