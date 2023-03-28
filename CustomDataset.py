# import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class CustomTensorDataset(Dataset):
    def __init__(self, X, Y, transform_list=None, train=True, train_fraction=0.7):
        # # [data_X, data_y] = dataset
        # X_tensor, y_tensor = torch.tensor(data_X), torch.tensor(data_y)
        #X_tensor, y_tensor = Tensor(data_X), Tensor(data_y)
        # self.train_fraction = train_fraction
        # self.is_train = train

        # x_train, x_test, y_train, y_test = train_test_split(
        #     X, Y, test_size=1-train_fraction, train_size=train_fraction, random_state=4)
        
        # tensors = None

        # if train:
        #     tensors = (x_train, y_train)
        # else:
        #     tensors = (x_test, y_test)

        tensors = (X, Y)

        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transforms = transform_list

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transforms:
            # for transform in self.transforms:
            #  x = transform(x)
            x = self.transforms(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)
