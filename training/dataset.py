# -*- coding: utf-8 -*-

import gc

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class EmbeddingsDataset(Dataset):
    def __init__(self, csv_path='train_dataset', transform=None):
        # Load data
        # Contains all data. Dict so we speedup reading from memory
        data = pd.read_pickle(csv_path)
        self.__data = {key: [values[0], values[1], values[2], values[3]] for key, values in enumerate(
            zip(data['filename'], data['video_embedding'], data['audio_embedding'], data['label']))}

        # Code for h5py version
        # self.path = csv_path

        # if shuffle:
        #     np.random.shuffle(self.__locs)
        # Use pos_weight value to overcome imbalanced dataset.
        pos_labels = np.array(data['label']).sum()
        # https://pytorch.org/docs/stable/nn.html#torch.nn.BCEWithLogitsLoss
        self.__pos_weight = (data['label'].shape[0] - pos_labels) / pos_labels
        # self.__pos_weight = 1.

        # Force garbage collector to get rid of the data
        data = None
        gc.collect()

        self.transform = transform

    @property
    def get_pos_weight(self):
        return self.__pos_weight

    def __len__(self):
        return len(self.__data.keys())
        # Code for h5py version
        # return 119154

    def __getitem__(self, idx):
        # Create sample
        # Code for h5py version
        # with h5py.File(self.path, mode='r') as f:
        #     sample = {
        #         'video_embedding': f['video_embedding'][idx].reshape((-1, 512)),
        #         'audio_embedding': f['audio_embedding'][idx],
        #         'label': f['label'][idx]
        #     }
        sample = {
                'filename': self.__data[idx][0],
                'video_embedding': self.__data[idx][1],
                'audio_embedding': self.__data[idx][2],
                'label': self.__data[idx][3]
        }

        # Transform (if defined)
        return self.transform(sample) if self.transform else sample


class RandomCrop:
    def __init__(self, crop_len):
        self.crop_len = crop_len

    def __call__(self, sample):
        video_embedding = sample['video_embedding']
        # Randomly choose an index
        tot_frames = video_embedding.shape[0]

        if self.crop_len > tot_frames:
            cropped_embedding = np.zeros((self.crop_len, video_embedding.shape[1]), dtype='float32')
            cropped_embedding[0:tot_frames, ...] = video_embedding
        else:
            start_idx = int(np.random.random() * (tot_frames - self.crop_len))
            end_idx = start_idx + self.crop_len
            cropped_embedding = video_embedding[start_idx:end_idx, ...]

        return {**sample, 'video_embedding': cropped_embedding}


class ToTensor:
    def __call__(self, sample):
        video_embedding = torch.tensor(sample['video_embedding']).float()
        audio_embedding = torch.tensor(sample['audio_embedding']).float()
        # audio_embedding = torch.tensor(np.zeros_like(sample['audio_embedding'])).float()
        label = torch.tensor(sample['label']).float()
        return {**sample, 'video_embedding': video_embedding, 'audio_embedding': audio_embedding, 'label': label}


class LabelOneHot:
    def __init__(self):
        self.l = {
            0: np.array([0, 1], dtype='uint8'),
            1: np.array([1, 0], dtype='uint8')
        }

    def __call__(self, sample):
        return {**sample, 'label': self.l[sample['label']]}


if __name__ == '__main__':

    # Initialize dataset
    dataset = EmbeddingsDataset('audio_video_embeddings.csv')

    # Test sampling
    sample = dataset[0]

    print('-----------------')
    print('--- EMBEDDING ---')
    print('-----------------')
    print(sample['embedding'])

    print('-----------------')
    print('----- LABEL -----')
    print('-----------------')
    print(sample['label'])

    # Test RandomCrop
    crop_len = 20
    rc = RandomCrop(crop_len)
    sample = rc(sample)

    # Test ToTensor
    tt = ToTensor()
    sample = tt(sample)

    # Test dataloader
    crop_len = 10
    trans = transforms.Compose([RandomCrop(crop_len),
                                ToTensor()
                                ])
    dataset = EmbeddingsDataset('audio_video_embeddings.csv', transform=trans)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for batch_sample in dataloader:
        batch = batch_sample['video_embedding']
        print(batch.shape)
        break
