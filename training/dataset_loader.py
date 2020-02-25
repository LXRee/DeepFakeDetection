# -*- coding: utf-8 -*-

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from functools import reduce
from torchvision import transforms
from typing import List, Dict, Tuple
import json
import pandas as pd


class EmbeddingsDataset(Dataset):
    def __init__(self, csv_path, shuffle=True, transform=None):
        # Load data
        # contains all data with indexes, which are accessed by df['embedding'].loc[int]
        # or df['label'].loc[int] method
        self.__data = pd.read_pickle(csv_path)

        # keep locations indexes without loading the values in memory
        self.__locs = np.linspace(0, self.__data.shape[0], self.__data.shape[0], dtype='uint32')

        if shuffle:
            np.random.shuffle(self.__locs)
        # Use pos_weight value to overcome imbalanced dataset.
        pos_labels = self.__data['label'][self.__data['label'] == 0].count()
        # https://pytorch.org/docs/stable/nn.html#torch.nn.BCEWithLogitsLoss
        self.__pos_weight = pos_labels / (self.__data.shape[0] - pos_labels)

        self.transform = transform

    def __len__(self):
        return self.__locs.shape[0]

    def __getitem__(self, idx):
        # Get data at index
        index = self.__locs[idx]

        # Create sample
        sample = {'embedding': self.__data['embedding'].loc[index], 'label': self.__data['label'].loc[index]}
        # Transform (if defined)
        if self.transform:
            sample = self.transform(sample)
        return sample


class RandomCrop:
    def __init__(self, crop_len):
        self.crop_len = crop_len

    def __call__(self, sample):
        embedding = sample['embedding']
        # Randomly choose an index
        tot_frames = len(embedding)
        start_idx = np.random.randint(0, tot_frames - self.crop_len)
        end_idx = start_idx + self.crop_len
        cropped_embedding = embedding[start_idx:end_idx, ...]
        return {**sample,
                'embedding': cropped_embedding
                }


class ToTensor:
    def __call__(self, sample):
        # Convert one hot encoded text to pytorch tensor
        embedding = torch.tensor(sample['embedding']).float()
        return {'embedding': embedding}


if __name__ == '__main__':

    # %% Initialize dataset
    filepath = 'dante_divina_commedia.txt'
    dataset = EmbeddingsDataset('embeddings/train_embeddings_complete.csv')

    # %% Test sampling
    sample = dataset[0]

    print('##############')
    print('##############')
    print('EMBEDDING')
    print('##############')
    print(sample['embedding'])

    print('##############')
    print('##############')
    print('LABEL')
    print('##############')
    print(sample['label'])

    # %% Test RandomCrop
    crop_len = 20
    rc = RandomCrop(crop_len)
    sample = rc(sample)

    # %% Test ToTensor
    tt = ToTensor()
    sample = tt(sample)

    # %% Test dataloader
    crop_len = 10
    trans = transforms.Compose([RandomCrop(crop_len),
                                ToTensor()
                                ])
    dataset = EmbeddingsDataset(
        'embeddings/train_embeddings_complete.csv',
        transform=trans
    )

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for batch_sample in dataloader:
        batch = batch_sample['embedding']
        print(batch.shape)
        break
