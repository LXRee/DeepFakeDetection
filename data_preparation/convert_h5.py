"""
Ho creato questo file nella speranza di trovare un formato più carino per il caricamento dei dati nel DataLoader.
Il caricamento lazy è bello - la RAM non è più saturata -  ma l'IO col disco è troppo lento poiché la quantità di dati
che deve recuperare per ogni indice è veramente poca. Quindi è preferibile rimanere col vecchio metodo pandas.
"""

import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm


def convert_pd_h5(csv_path, h5_path):
    df = pd.read_pickle(csv_path)
    len_df = df['label'].shape[0]
    h5f = h5py.File(h5_path, mode='w')
    # define special datatype to store different length array for video embeddings and strings
    dt_string = h5py.string_dtype(encoding='utf-8')
    dt_video = h5py.special_dtype(vlen=np.dtype('float32'))
    h5f.create_dataset('filename', shape=(len_df,), dtype=dt_string)
    h5f.create_dataset('video_embedding', shape=(len_df,), dtype=dt_video)
    h5f.create_dataset('audio_embedding', shape=(len_df, 50), dtype='float32')
    h5f.create_dataset('label', shape=(len_df,), dtype='uint8')

    for i in tqdm(range(len_df)):
        filename = df['filename'].loc[i]
        video_emb_flatten = df['video_embedding'].loc[i].flatten()
        audio_emb = df['audio_embedding'].loc[i]
        label = df['label'].loc[i]
        h5f['filename'][i] = filename
        h5f['video_embedding'][i] = video_emb_flatten
        h5f['audio_embedding'][i] = audio_emb
        h5f['label'][i] = label
    h5f.close()


if __name__ == '__main__':
    csv_path = 'dataset/train_audio_video_embeddings.csv'
    h5_path = 'dataset/train_audio_video_embeddings.h5'
    convert_pd_h5(csv_path, h5_path)
