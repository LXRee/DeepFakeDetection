import gc
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class EmbeddingsDataset(Dataset):
    def __init__(self, real_csv_path='train_dataset', fake_csv_path='train_dataset', transform=None):
        # Load data
        # Contains all data. Dict so we speedup reading from memory
        real_data = pd.read_pickle(real_csv_path)
        fake_data = pd.read_pickle(fake_csv_path)
        # Read fake and real video information
        self.__real_data = {
            key: [np.array(values[0]) if isinstance(values[0], list) else values[0], values[1]]
            for key, values in enumerate(
                zip(real_data['video_embedding'], real_data['audio_embedding']))}
        self.__fake_data = {
            key: [np.array(values[0]) if isinstance(values[0], list) else values[0], values[1]]
            for key, values in enumerate(
                zip(fake_data['video_embedding'], fake_data['audio_embedding']))}

        # Keep list of integers in order to mix couples when calling getitem
        self.__prev_index = 0

        gc.collect()

        self.transform = transform

    def __len__(self):
        return len(self.__real_data.keys())

    def __getitem__(self, idx):
        real_video_embedding, real_audio_embedding = self.__real_data[self.__prev_index]
        fake_video_embedding, fake_audio_embedding = self.__fake_data[idx]
        self.__prev_index = idx
        # Return sample and its label, with video embeddings and audio_embeddings in this order
        #   [0, 1] -> real, fake
        #   [1, 0] -> fake, real
        if random.random() > 0.5:
            sample = \
                real_video_embedding, \
                real_audio_embedding, \
                fake_video_embedding, \
                fake_audio_embedding, \
                0.
        else:
            sample = \
                fake_video_embedding, \
                fake_audio_embedding, \
                real_video_embedding, \
                real_audio_embedding, \
                1.

        # Transform (if defined)
        return self.transform(sample) if self.transform else sample


class RandomCrop:
    def __init__(self, crop_len):
        self.crop_len = crop_len

    def crop(self, embedding):
        # Randomly choose an index
        embedding = embedding[::3, ...]

        tot_frames = embedding.shape[0]

        if self.crop_len > tot_frames:
            cropped_embedding = np.zeros((self.crop_len, embedding.shape[1]), dtype='float32')
            cropped_embedding[0:tot_frames, ...] = embedding
        else:
            start_idx = int(np.random.random() * (tot_frames - self.crop_len))
            end_idx = start_idx + self.crop_len
            cropped_embedding = embedding[start_idx:end_idx, ...]
        return cropped_embedding

    def __call__(self, sample):
        video_embedding_0, audio_embedding_0, video_embedding_1, audio_embedding_1, label = sample
        cropped_video_embedding_0 = self.crop(video_embedding_0)
        cropped_video_embedding_1 = self.crop(video_embedding_1)
        return cropped_video_embedding_0, audio_embedding_0, cropped_video_embedding_1, audio_embedding_1, label


class ToTensor:
    def __call__(self, sample):
        return list(torch.tensor(t).float() for t in sample)
