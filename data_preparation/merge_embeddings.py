import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def merge_dataframes(source, dest):
    """
    Takes all dataframes in source folder and merge into a single Dataframe saved in dest folder
    :param source: source folder of the embeddings
    :param dest: destination folder for the embeddings
    """
    df = pd.DataFrame(columns=['filename', 'embedding', 'label'])
    for root, dirs, files in os.walk(source):
        for file in tqdm(files, desc="Reading source"):
            df_part = pd.read_pickle(os.path.join(root, file))
            df = pd.concat([df, df_part])

    initial_df_dim = df['label'].shape[0]

    # Sanity checks
    deletion_indexes = []
    seq_threshold = [1, 2, 4, 6]
    no_embeddings = 0
    small_seq = 0
    for e in tqdm(range(df['embedding'].shape[0]), desc='Sanity checks'):
        # Check if every embedding has 2 dimensions
        if len(df['embedding'].loc[e].shape) is not 2:
            deletion_indexes.append(e)
            no_embeddings += 1

        # Check if the sequence dimension is too small (above 3 -> 4*6 (window) = 24 frames, at least 1 sec video)
        seq_dim = df['embedding'].loc[e].shape[0]
        if seq_dim < seq_threshold[0]:
            deletion_indexes.append(e)
            small_seq += 1

    print("Found {} None embeddings and {} too small sequences".format(no_embeddings, small_seq))
    deletion_indexes.sort()
    for index in tqdm(deletion_indexes, desc="Deletion of bad rows"):
        df = df.drop(df.index[index])

    df = df.reset_index(drop=True)
    final_df_dim = df['label'].shape[0]
    print("Initial df dimension: {}\n"
          "Final df dimension after sanity checks: {}\n"
          "Now saving...".format(initial_df_dim, final_df_dim))
    df.to_pickle(os.path.join(dest, 'merged.csv'))
    print('Merge saved at file: {}'.format(os.path.join(dest, 'merged.gzip')))


def merge_audio_video_df(audio_df, video_df, for_submission=False):
    """
    All audio_df and video_df filenames are unique, so we are using this information to store a key-value dict to
    index audio_df and load its information in video_df
    :param audio_df: dataframe containing audio embeddings with (filename, audio_embedding) columns
    :param video_df: dataframe containing video embeddings with (filename, video_embedding, label) columns
    :return: new dataframe with (filename, video_embedding, audio_embedding, label) columns
    """
    # Create reverse mapping for audio dataframe (from filename to index). All filenames are unique.
    label_to_key = {}
    for i in range(len(audio_df['filename'])):
        label_to_key[audio_df['filename'].loc[i]] = i

    # Create empty dataframe
    if for_submission:
        df = pd.DataFrame(columns=['filename', 'video_embedding', 'audio_embedding'])
    else:
        df = pd.DataFrame(columns=['filename', 'video_embedding', 'audio_embedding', 'label'])

    video_no_audio = 0

    for i in tqdm(range(len(video_df['filename'])), desc="Merging progress"):
        file_path = video_df['filename'].loc[i]
        file_name = os.path.basename(file_path).split('.')[0]  # in case the name is a path
        try:
            audio_embedding = audio_df['audio_embedding'].loc[label_to_key[file_name]]
        except KeyError:
            # In case the video has no audio, we put a ones vector of dimension 50.
            # I got the dimension hardcoded from class "MASRCNN_activate" in "create_audio_embeddings",
            # num_dense_neurons. It has to be changed if changed also there.
            audio_embedding = np.ones(50, dtype='float32')
            video_no_audio += 1
            print('{} has no audio! Counted {}'.format(file_path, video_no_audio))
        if for_submission:
            df.loc[i] = [file_path, video_df['embedding'].loc[i], audio_embedding]
        else:
            df.loc[i] = [file_path, video_df['embedding'].loc[i], audio_embedding, video_df['label'].loc[i]]

    return df


def save_many_in_path(path, df):
    for i in tqdm(range(len(df['label'])), desc='Saving files in {}'.format(path)):
        filename = os.path.basename(df['filename'].loc[i]).split('.')[0]
        video_embedding = df['video_embedding'].loc[i]
        audio_embedding = df['audio_embedding'].loc[i]
        label = df['label'].loc[i]
        new_df = pd.DataFrame(columns=['filename', 'video_embedding', 'audio_embedding', 'label'])
        new_df.loc[0] = [filename, video_embedding, audio_embedding, label]
        new_df.to_pickle(os.path.join(path, filename + '.csv'))


if __name__ == '__main__':
    # merge_dataframes('video_embeddings/partials', 'video_embeddings')
    print("Loading video and audio DataFrames...")
    audio_df = pd.read_pickle('test_audio_embeddings.csv')
    video_df = pd.read_pickle('test_video_embeddings.csv')
    print("DataFrames loaded. Now merging...")
    df = merge_audio_video_df(audio_df, video_df, for_submission=False)
    print("Finish merging. Now saving...")
    df.to_pickle('test_audio_video_embeddings.csv')
