import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import classification_report
from CustomDataset import CustomTensorDataset
from torch.utils.data import DataLoader
from utils import get_images_and_targets


from transformers import AutoFeatureExtractor, AutoModelForImageClassification


from lightning_model import LightningModel

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

CHECKPOINT_MODEL_PATH = './output/model/resnet-50-3dprint-1e-06-100-epoch=29-val_f1=0.8055_cv.ckpt'


def main():

    print("loading train and test data")
    train_labels = pd.read_csv(train_set)
    test_labels = pd.read_csv(test_set)


    image_processor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)

    valX, valY = get_images_and_targets(train_labels, images, image_processor, test=False, val=True, train_fraction=0.8)
    dataset_val = CustomTensorDataset(valX, valY)
    valloader = DataLoader(dataset=dataset_val, batch_size=100, shuffle=True, num_workers=6)

    testX, imgpathsY = get_images_and_targets(test_labels, images, image_processor, test=True)
    dataset_test = CustomTensorDataset(testX, torch.ones(len(imgpathsY)).reshape(-1,1))
    testloader = DataLoader(dataset=dataset_test, batch_size=100, shuffle=False, num_workers=6)

    pretrained_model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={str(i): c for i, c in enumerate(range(2))},
        label2id={c: str(i) for i, c in enumerate(range(2))},
        ignore_mismatched_sizes = True,
    )


    model = LightningModel(pretrained_model, num_classes=2,reshape_input=False).load_from_checkpoint(CHECKPOINT_MODEL_PATH, model=pretrained_model, num_classes=2)


    device = torch.device('cuda:0')
    model = model.to(device)

    softmax = torch.nn.Softmax(dim=1)

    # for evaluating performance on validation set
    gt_max, pred_max, probs_all = [], [], []
    for loader in [valloader]:
        with torch.no_grad():

            for idx, data in tqdm(enumerate(loader)):

                img_seq, label = data
                img_seq = img_seq.cuda(device)
                logits = model(img_seq)


                probs = softmax(logits)
                preds = torch.max(probs, 1, keepdim=True)[1].int().cpu()
                
                
                gt_max.append(label)
                pred_max.append( preds)
                probs_all.append(probs)


        gt_max = torch.vstack(gt_max).cpu()
        pred_max = torch.vstack(pred_max).cpu()
        probs_all = torch.vstack(probs_all).cpu()

        print(classification_report(gt_max, pred_max, zero_division=0, digits=7))


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
    results_df.to_csv(SAVING_OUTPUTS+f'results_{saving_name}_0000001_epoch_29_cv_f1.csv', index=False)


if __name__ =='__main__':
    main()