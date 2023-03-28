import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

class LightningModel(pl.LightningModule):
    """
    Generic lightning model to encapsulate specific models for training. 
    """

    def __init__(self, model, num_classes, learning_rate=1e-5, optimizer='adam', reshape_input=True):
        super(LightningModel, self).__init__()
        
        self.model = model
        self.n_class = num_classes
        self.lr = learning_rate

        self.opt = optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        outs = self.model.forward(x).logits
            
        return outs

    def training_step(self, batch, batch_idx):
        img_seq, label = batch
        gt_label = F.one_hot(label.view(1, -1)[0].long(), num_classes=self.n_class).float()
        
        # if self.reshape_input:
        #     img_seq = torch.permute(img_seq, (0,2,1,3,4))

        logits = self.forward(img_seq) # output size: batch_size x 4
        probs = self.softmax(logits)
        preds = torch.max(probs, 1, keepdim=True)[1].int()
        
        # print(logits.size(), probs.size(), gt_label.size())
        
        loss = self.criterion(logits, gt_label)


        # print(probs, label, label.long().reshape(-1))
        # prlong(label.long(), label.long().reshape(-1))
        # print(label.size(), probs.size(), preds.size(), torch.max(probs, 1, keepdim=True).size(), len(batch))
        # top1acc = torchmetrics.functional.accuracy(preds, label, task='binary')
        f1 = torchmetrics.functional.f1_score(preds, label, task='binary', average='weighted')

        # self.log('train_tm', , prog_bar=True)

        self.log('train_f1_batch', f1, prog_bar=True)
        # self.train_accuracy.update(probs, label.long().reshape(-1))
        # self.train_accuracy_3.update(probs, label.long().reshape(-1))
        # self.train_accuracy_5.update(probs, label.long().reshape(-1))
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
        # if self.reshape_input:
        #     img_seq = torch.permute(img_seq, (0,2,1,3,4))
        
        logits = self.forward(img_seq) # output size: batch_size x 49
        probs = self.softmax(logits)
        preds = torch.max(probs, 1, keepdim=True)[1].int()
        

        # top1acc = torchmetrics.functional.accuracy(preds, label, task='binary')
        f1 = torchmetrics.functional.f1_score(preds, label, task='binary', average='weighted')
        return {'f1': f1}



    
    def validation_epoch_end(self, outputs):

        top1 = sum(output['f1'] for output in outputs) / len(outputs)
        self.logger.experiment.add_scalar("F1/Validation", top1, self.current_epoch)
        self.log('val_f1', top1, prog_bar=True)

    

    def predict_step(self, batch, batch_idx):
        img_seq, label = batch
        # img_seq = torch.permute(img_seq, (0,2,1,3,4))
        logits = self.forward(img_seq)
        probs = self.softmax(logits)

        preds = torch.max(probs, 1, keepdim=True)[1].int()
        return probs, preds

    def configure_optimizers(self):
        
        return torch.optim.SGD(self.parameters(), lr=self.lr) if self.opt =='sgd' else torch.optim.Adam(self.parameters(), lr=self.lr)
    