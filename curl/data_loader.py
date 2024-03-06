from random import random
import numpy as np
from copy import copy

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data, Dataset
import torch
from custom_dataloader import GeneralDataset
import collections


DatasetTuple = collections.namedtuple('DatasetTuple', [
    'train_dataloader', 'train_dataloader_for_clf',
    'valid_dataloader', 'test_dataloader', "n_classes", "output_shape",
    "num_train_examples"
])

class GeneralDataset(Dataset):
    def __init__(self, data, targets, transform = None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        if torch.is_tensor(self.data):
            return self.data.size(dim = 0)
        elif isinstance(self.data, np.ndarray):
            return self.data.shape[0]
        return len(self.data)

    def __getitem__(self, idx):
        input_features = self.data[idx]
        if self.transform is not None:
            input_features = self.transform(input_features)
        target = self.targets[idx]
        return input_features, target





def filter_dataset(filter_fn, dataset):
    idx = np.where(filter_fn(dataset.targets))[0].tolist()
    accessed_targets = map(dataset.targets.__getitem__, idx)
    accessed_data = map(dataset.data.__getitem__, idx)
    new_dataset = GeneralDataset(list(accessed_data), list(accessed_targets))
    return new_dataset

def binarize_fn(x):
    """Binarize a Bernoulli by rounding the probabilities.

    Args:
      x: torch tensor, input image.

    Returns:
      A torch tensor with the binarized image
    """
    return torch.gt(x, 0.5 * torch.ones_like(x)).float()


def data_loader_MNIST(training_data_type, n_concurrent_classes, root_dir = './data', BATCH_SIZE = 1, EVAL_BATCH_SIZE = 1):


    transform = transforms.Compose([transforms.ToTensor()])
                                    # normalization values = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
    #trainset = datasets.MNIST(root = '../data', train = True, download = True, transform = transform)
    #testset  = datasets.MNIST(root = '../data', train = False, download = True, transform = transform)
    trainset = torch.load('/home/gridsan/aejilemele/LIS/SHELS-main/datasets/mnist/trainset.pt')
    testset = torch.load('/home/gridsan/aejilemele/LIS/SHELS-main/datasets/mnist/testset.pt')

    train_data_len = int(0.88*len(trainset))
    val_data_len = len(trainset) - train_data_len
    print(type(trainset.data[0]), 'type of trainset data after doing create traintestset mnist')
    train_set, val_set = data.random_split(trainset,[train_data_len, val_data_len])

    train_indices = train_set.indices
    val_indices = val_set.indices
    if torch.is_tensor(train_set.dataset.data[0]):
        binarize_transform = transforms.compose([transforms.Lambda(lambda x: {'image': binarize_fn(x)})()])
        trainset = GeneralDataset(np.array(train_set.dataset.data)[train_indices].tolist(), np.array(train_set.dataset.targets)[train_indices], transform=binarize_transform)
        valset = GeneralDataset(np.array(val_set.dataset.data)[val_indices].tolist(), np.array(val_set.dataset.targets)[val_indices], transform = binarize_transform)
    else:
        binarize_transform = transforms.compose([transforms.ToTensor(), transforms.Lambda(lambda x: {'image': binarize_fn(x)})()])
        trainset = GeneralDataset(np.array(train_set.dataset.data)[train_indices].tolist(), np.array(train_set.dataset.targets)[train_indices], transform=binarize_transform)
        valset = GeneralDataset(np.array(val_set.dataset.data)[val_indices].tolist(), np.array(val_set.dataset.targets)[val_indices], transform = binarize_transform)

    if torch.is_tensor(testset.data[0]):
        binarize_transform = transforms.compose([transforms.Lambda(lambda x: {'image': binarize_fn(x)})()])
        testset = GeneralDataset(testset.data, testset.targets, transform = binarize_transform )
    else:
        binarize_transform = transforms.compose([transforms.ToTensor(), transforms.Lambda(lambda x: {'image': binarize_fn(x)})()])
        testset = GeneralDataset(testset.data, testset.targets, transform = binarize_transform )


    n_classes = len(trainset.classes)
    num_train_examples = len(trainset.data)
    output_shape = trainset.data[0].shape

    
    if training_data_type == 'sequential':
        c = None
        if n_concurrent_classes == 1: # datapoint tuple of x, y
            filter_fn = lambda datapoint: torch.equal(datapoint[1], c)
        else:
            # Define the lowest and highest class number at each data period.
            assert n_classes % n_concurrent_classes == 0, (
                'Number of total classes must be divisible by '
                'number of concurrent classes')
            cmin = []
            cmax = []
            for i in range(int(n_classes / n_concurrent_classes)):
                for _ in range(n_concurrent_classes):
                    cmin.append(i * n_concurrent_classes)
                    cmax.append((i + 1) * n_concurrent_classes)
            
            filter_fn = lambda y: torch.logical_and(
            torch.greater_equal(y, cmin[c]), torch.less(y, cmax[c]))

        
        train_datasets = []
        train_dataloaders = []

        for c in range(n_classes):
            filtered_ds = filter_dataset(filter_fn, trainset)
            train_datasets.append(filtered_ds)
            train_dataloaders.append(data.DataLoader(filtered_ds, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2, drop_last = True))

    else: # not an sequential 
        train_datasets = trainset
        train_dataloaders  = data.DataLoader(train_datasets, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2, drop_last = True)

    
    train_dataset_for_classifier = trainset
    train_dataloader_for_classifier = data.DataLoader(train_dataset_for_classifier, batch_size = EVAL_BATCH_SIZE, shuffle = True, num_workers = 2, drop_last = True)
    


    ## deal with the val set
    num_valid_examples = len(valset)
    assert (num_valid_examples %
            EVAL_BATCH_SIZE == 0), ('EVAL_BATCH_SIZE must be a divisor of %d' %
                                    num_valid_examples)
    valid_dataloader = data.DataLoader(valset, batch_size = EVAL_BATCH_SIZE, shuffle = True, num_workers = 2, drop_last = True)

    num_test_examples = len(testset)
    assert (num_test_examples %
            EVAL_BATCH_SIZE == 0), ('EVAL_BATCH_SIZE must be a divisor of %d' %
                                    num_test_examples)
    test_dataloader = data.DataLoader(testset, batch_size = EVAL_BATCH_SIZE, shuffle = True, num_workers = 2, drop_last = True)


    return DatasetTuple(train_dataloaders, train_dataloader_for_classifier,
                      valid_dataloader,
                      test_dataloader, n_classes, output_shape, num_train_examples )



def load_data(dataset, training_data_type, n_concurrent_classes, BATCH_SIZE = 1, EVAL_BATCH_SIZE = 1):
    if dataset == "mnist":
        return data_loader_MNIST(training_data_type, n_concurrent_classes,BATCH_SIZE, EVAL_BATCH_SIZE, root_dir = './data',  )