import numpy as np
import torch
import torchmetrics
import torch.nn.functional as F
import pytorch_lightning as pl

        
class ConvNet(pl.LightningModule):
    
    def __init__(self, n_inputs, n_hiddens_per_conv_layer, n_hiddens_per_fc_layer, n_outputs, 
                 patch_size_per_conv_layer, stride_per_conv_layer, activation_function='tanh', learning_rate=0.01, optimizer='adam'):
        
        super().__init__()
        
        # self.device = device

        n_conv_layers = len(n_hiddens_per_conv_layer)
        if (len(patch_size_per_conv_layer) != n_conv_layers or
            len(stride_per_conv_layer) != n_conv_layers):
            raise Exception('The lengths of n_hiddens_per_conv_layer, patch_size_per_conv_layer, and stride_per_conv_layer must be equal.')
        
        self.activation_function = torch.tanh if activation_function == 'tanh' else torch.relu
        
        # Create all convolutional layers
        # First argument to first Conv2d is number of channels for each pixel.
        # Just 1 for our grayscale images.
        n_in = 3
        input_hw = int(np.sqrt(n_inputs))  # original input image height (=width because image assumed square)
        self.conv_layers = torch.nn.ModuleList()
        for nh, patch_size, stride in zip(n_hiddens_per_conv_layer,
                                          patch_size_per_conv_layer,
                                          stride_per_conv_layer):
            self.conv_layers.append( torch.nn.Conv2d(n_in, nh, kernel_size=patch_size, stride=stride) )
            conv_layer_output_hw = (input_hw - patch_size) // stride + 1
            input_hw = conv_layer_output_hw  # for next trip through this loop
            n_in = nh

        # Create all fully connected layers.  First must determine number of inputs to first
        # fully-connected layer that results from flattening the images coming out of the last
        # convolutional layer.
        n_in = input_hw ** 2 * n_in  # n_hiddens_per_fc_layer[0]
        self.fc_layers = torch.nn.ModuleList()
        for nh in n_hiddens_per_fc_layer:
            self.fc_layers.append( torch.nn.Linear(n_in, nh) )
            n_in = nh
        self.fc_layers.append( torch.nn.Linear(n_in, n_outputs) )


        self.n_class = n_outputs
        self.lr = learning_rate

        self.opt = optimizer
        self.criterion = torch.nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax(dim=1)
        
        # self.to(self.device)


    def forward_all_outputs(self, X):
        n_samples = X.shape[0]
        Ys = [X]
        for conv_layer in self.conv_layers:
            Ys.append( self.activation_function(conv_layer(Ys[-1])) )

        for layeri, fc_layer in enumerate(self.fc_layers[:-1]):
            if layeri == 0:
                Ys.append( self.activation_function(fc_layer(Ys[-1].reshape(n_samples, -1))) )
            else:
                Ys.append( self.activation_function(fc_layer(Ys[-1])) )

        Ys.append(self.fc_layers[-1](Ys[-1]))
        return Ys


    def forward(self, X):
        Ys = self.forward_all_outputs(X)
        return Ys[-1]
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
        gt_label = F.one_hot(label.view(1, -1)[0].long(), num_classes=self.n_class).float()
        # if self.reshape_input:
        #     img_seq = torch.permute(img_seq, (0,2,1,3,4))
        
        logits = self.forward(img_seq) # output size: batch_size x 49
        probs = self.softmax(logits)
        preds = torch.max(probs, 1, keepdim=True)[1].int()

        loss = self.criterion(logits, gt_label)
        

        # top1acc = torchmetrics.functional.accuracy(preds, label, task='binary')
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
        # img_seq = torch.permute(img_seq, (0,2,1,3,4))
        logits = self.forward(img_seq)
        probs = self.softmax(logits)

        preds = torch.max(probs, 1, keepdim=True)[1].int()
        return probs, preds

    def configure_optimizers(self):
        
        return torch.optim.SGD(self.parameters(), lr=self.lr) if self.opt =='sgd' else torch.optim.Adam(self.parameters(), lr=self.lr)
    

    # def train(self, X, T, batch_size, n_epochs, learning_rate, method='sgd', verbose=True):
        
    #     # Set data matrices to torch.tensors if not already.
    #     if not isinstance(X, torch.Tensor):
    #         X = torch.from_numpy(X).float().to(self.device)
    #     if not isinstance(T, torch.Tensor):
    #         T = torch.from_numpy(T).long().to(self.device)  # required for classification in pytorch
            
    #     X = torch.permute(X, (0, 3, 1, 2))
    #     X.requires_grad_(True)
        
    #     self.classes = torch.unique(T)
        
    #     if method == 'sgd':
    #         optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
    #     else:
    #         optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    #     CELoss = torch.nn.CrossEntropyLoss(reduction='mean')
    #     self.error_trace = []
        
    #     for epoch in range(n_epochs):

    #         num_batches = X.shape[0] // batch_size
    #         loss_sum = 0
            
    #         for k in range(num_batches):
    #             start = k * batch_size
    #             end = (k + 1) * batch_size
    #             X_batch = X[start:end, ...]
    #             T_batch = T[start:end, ...]
                
    #             Y = self.forward(X_batch)

    #             loss = CELoss(Y, T_batch)
    #             loss.backward()

    #             # Update parameters
    #             optimizer.step()
    #             optimizer.zero_grad()

    #             loss_sum += loss

    #         self.error_trace.append(loss_sum / num_batches)

    #         if verbose and (epoch + 1) % (n_epochs // 10) == 0:
    #             print(f'{method}: Epoch {epoch + 1} Loss {self.error_trace[-1]:.3f}')

    #     return self


    # def softmax(self, Y):
    #     '''Apply to final layer weighted sum outputs'''
    #     # Trick to avoid overflow
    #     maxY = torch.max(Y, axis=1)[0].reshape((-1,1))
    #     expY = torch.exp(Y - maxY)
    #     denom = torch.sum(expY, axis=1).reshape((-1, 1))
    #     Y = expY / denom
    #     return Y


    def use(self, X):
        # Set input matrix to torch.tensors if not already.
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).float()
        
        X = torch.permute(X, (0, 3, 1, 2))
        Y = self.forward(X)
        probs = self.softmax(Y)
        classes = self.classes[torch.argmax(probs, axis=1).cpu().numpy()]
        return classes.cpu().numpy(), probs.detach().cpu().numpy()
