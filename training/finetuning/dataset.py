import gc

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class EmbeddingsDataset(Dataset):
    def __init__(self, real_csv_path='train_dataset', fake_csv_path='fake_dataset', transform=None):
        # Load data
        # Contains all data. Dict so we speedup reading from memory
        print("Reading pickled csv files...")
        real_data = pd.read_pickle(real_csv_path)
        fake_data = pd.read_pickle(fake_csv_path)
        self.__data = {}
        k = 0
        for values in zip(fake_data['filename'], fake_data['video_embedding'], fake_data['audio_embedding'], fake_data['label']):
            self.__data[k] = [values[0], np.array(values[1]) if isinstance(values[1], list) else values[1], values[2], values[3]]
            k += 1
        for values in zip(real_data['filename'], real_data['video_embedding'], real_data['audio_embedding'], real_data['label']):
            self.__data[k] = [values[0], np.array(values[1]) if isinstance(values[1], list) else values[1], values[2], values[3]]
            k += 1

        # Code for h5py version
        # self.path = csv_path

        # if shuffle:
        #     np.random.shuffle(self.__locs)
        # Use pos_weight value to overcome imbalanced dataset.
        # pos_labels = np.array(self.__data['label']).sum()
        # https://pytorch.org/docs/stable/nn.html#torch.nn.BCEWithLogitsLoss
        # self.__pos_weight = (self.__data['label'].shape[0] - pos_labels) / pos_labels
        self.__pos_weight = 1.

        # Force garbage collector to get rid of the data
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
        sample = self.__data[idx]
        # Transform (if defined)
        return self.transform(sample) if self.transform else sample


class RandomCrop:
    def __init__(self, crop_len):
        self.crop_len = crop_len

    def __call__(self, sample):
        filename, video_embedding, audio_embedding, label = sample
        # Randomly choose an index
        video_embedding = video_embedding[::3, ...]
        tot_frames = video_embedding.shape[0]

        if self.crop_len > tot_frames:
            cropped_embedding = np.zeros((self.crop_len, video_embedding.shape[1]), dtype='float32')
            cropped_embedding[0:tot_frames, ...] = video_embedding
        else:
            start_idx = int(np.random.random() * (tot_frames - self.crop_len))
            end_idx = start_idx + self.crop_len
            cropped_embedding = video_embedding[start_idx:end_idx, ...]

        return filename, cropped_embedding, audio_embedding, label


class ToTensor:
    def __call__(self, sample):
        return (sample[0], *[torch.tensor(s).float() for s in sample[1:]])


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
