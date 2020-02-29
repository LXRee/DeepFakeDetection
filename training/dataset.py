# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class EmbeddingsDataset(Dataset):
    def __init__(self, csv_path='train_dataset', shuffle=True, transform=None):
        # Load data
        # contains all data with indexes, which are accessed by df['embedding'].loc[int]
        # or df['label'].loc[int] method
        # path_list = []
        # for root, dirs, files in os.walk(csv_path):
        #     path_list.extend([os.path.join(root, file) for file in files])
        # self.__path_list = path_list
        data = pd.read_pickle(csv_path)

        # Keep locations indexes to better manage the files
        # self.__locs = np.linspace(0, data.shape[0] - 1, data.shape[0], dtype='uint32')
        self.__video_embeddings = list(data['video_embedding'])
        self.__audio_embeddings = list(data['audio_embedding'])
        self.__labels = np.array(data['label'])
        del data

        # if shuffle:
        #     np.random.shuffle(self.__locs)
        # Use pos_weight value to overcome imbalanced dataset.
        # pos_labels = self.__labels.sum()
        # https://pytorch.org/docs/stable/nn.html#torch.nn.BCEWithLogitsLoss
        # self.__pos_weight = (self.__labels.shape[0] - pos_labels) / pos_labels
        self.__pos_weight = 1.
        # self.reader = pd.read_pickle

        self.transform = transform

    @property
    def get_pos_weight(self):
        return self.__pos_weight

    def __len__(self):
        return self.__labels.shape[0]
        # return len(self.__path_list)

    def __getitem__(self, idx):
        # Get data at index
        # index = self.__locs[idx]
        # path_to_csv = self.__path_list[idx]
        # df = self.reader(path_to_csv)

        # Create sample
        sample = {
            'video_embedding': self.__video_embeddings[idx],
            'audio_embedding': self.__audio_embeddings[idx],
            'label': self.__labels[idx]
        }
        # sample = {'video_embedding': df['video_embedding'].loc[0],
        #           'audio_embedding': df['audio_embedding'].loc[0],
        #           'label': df['label'].loc[0]}
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
