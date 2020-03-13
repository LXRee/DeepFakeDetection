import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import shuffle


# def augment_not_fake(source, dest, min_length=9):
#     """
#     Doubles the enough-long video embeddings for not fake ones.
#     :param source: path to source dataframe
#     :param dest: path to destination dataframe
#     :param min_length: min embedding length
#     :return:
#     """
#     df = pd.read_pickle(source)
#     doubled_counter = 0
#     not_fake = 0
#     for i in tqdm(range(df['label'].shape[0])):
#         # look for true labels
#         if df['label'].loc[i] == 0:
#             temp_video_embedding = df['video_embedding'].loc[i]
#             if temp_video_embedding.shape[0] > min_length:
#                 df['video_embedding'].loc[i] = temp_video_embedding[0:round(temp_video_embedding.shape[0] / 2)]
#                 new_pos = df['label'].shape[0]
#                 df.loc[new_pos] = [df['filename'].loc[i] + '_0',
#                                                 temp_video_embedding[round(temp_video_embedding.shape[0] / 2):],
#                                                 df['audio_embedding'].loc[i],
#                                                 df['label'].loc[i]]
#                 doubled_counter += 1
#             not_fake += 1
#     print('Doubled {} not fake videos out of {}'.format(doubled_counter,
#                                                         not_fake))
#     df.to_pickle(dest)
#
#
# def merge_dataframes(source, dest):
#     """
#     Takes all dataframes in source folder and merge into a single Dataframe saved in dest folder
#     :param source: source folder of the embeddings
#     :param dest: destination folder for the embeddings
#     """
#     df = pd.DataFrame(columns=['filename', 'embedding', 'label'])
#     for root, dirs, files in os.walk(source):
#         for file in tqdm(files, desc="Reading source"):
#             df_part = pd.read_pickle(os.path.join(root, file))
#             df = pd.concat([df, df_part])
#
#     initial_df_dim = df['label'].shape[0]
#
#     # Sanity checks
#     deletion_indexes = []
#     seq_threshold = [1, 2, 4, 6]
#     no_embeddings = 0
#     small_seq = 0
#     for e in tqdm(range(df['embedding'].shape[0]), desc='Sanity checks'):
#         # Check if every embedding has 2 dimensions
#         if len(df['embedding'].loc[e].shape) is not 2:
#             deletion_indexes.append(e)
#             no_embeddings += 1
#
#         # Check if the sequence dimension is too small (above 3 -> 4*6 (window) = 24 frames, at least 1 sec video)
#         seq_dim = df['embedding'].loc[e].shape[0]
#         if seq_dim < seq_threshold[0]:
#             deletion_indexes.append(e)
#             small_seq += 1
#
#     print("Found {} None embeddings and {} too small sequences".format(no_embeddings, small_seq))
#     deletion_indexes.sort()
#     for index in tqdm(deletion_indexes, desc="Deletion of bad rows"):
#         df = df.drop(df.index[index])
#
#     df = df.reset_index(drop=True)
#     final_df_dim = df['label'].shape[0]
#     print("Initial df dimension: {}\n"
#           "Final df dimension after sanity checks: {}\n"
#           "Now saving...".format(initial_df_dim, final_df_dim))
#     df.to_pickle(os.path.join(dest, 'merged.csv'))
#     print('Merge saved at file: {}'.format(os.path.join(dest, 'merged.gzip')))


def merge_audio_video_embeddings(metadata_path, audio_embedding_path, video_embedding_path, dest_path):
    """
    Merge all audio and video embeddings into an unique dataframe.
    :param metadata_path: path/to/metadata.json file containing the list of all videos that has been processed.
    :param audio_embedding_path: path/to/folder containing all the audio embeddings for videos in metadata.json
    :param video_embedding_path: path/to/folder containing all the video embeddings for videos in metadata.json
    :param dest_path: path/to/dest/file for the embeddings
    :return:
    """
    # Create dict to convert labels to int values
    label_to_int = {'FAKE': 1, 'REAL': 0}
    # Retrieve metadata file used for video and audio extraction. Then create the list of path/to/videos
    metadata = json.load(open(metadata_path, 'r'))
    video_paths = list(metadata.keys())
    # Shuffle video paths since the metadata contains all REAL videos and the all FAKE ones.
    shuffle(video_paths)
    # Create new dataframe
    df = pd.DataFrame(columns=['filename', 'video_embedding', 'audio_embedding', 'label'])
    # Iterate over all video paths
    for path in tqdm(video_paths, desc="Merging embeddings"):
        # csv Filename
        filename = os.path.basename(path).split('.')[0] + '.csv'
        # Audio embedding
        try:
            audio_embedding = pd.read_pickle(os.path.join(audio_embedding_path, filename))['audio_embedding'].loc[0]
        except FileNotFoundError as e:
            audio_embedding = np.zeros((256, ), 'float32')
            print("I didn't find {}".format(e))
        # Video embedding
        try:
            video_embedding = pd.read_pickle(os.path.join(video_embedding_path, filename))['video_embedding'].loc[0]
        except FileNotFoundError as e:
            video_embedding = np.zeros((1, 512), 'float32')
            print("I didn't find {}".format(e))
        # Convert label directly from metadata
        label = label_to_int[metadata[path]['label']]
        # Append new row at the end of the dataframe
        df.loc[df['label'].shape[0]] = [os.path.basename(path), video_embedding, audio_embedding, label]
    # Save dataframe to disk
    df.to_pickle(dest_path)


if __name__ == '__main__':
    metadata_path = os.path.join('data', 'train_data', 'balanced_metadata.json')
    audio_embeddings_path = os.path.join('dataset', 'audio_embeddings')
    video_embeddings_path = os.path.join('dataset', 'video_embeddings')
    dest_path = os.path.join('dataset', 'train_audio_video_embeddings.csv')
    merge_audio_video_embeddings(metadata_path, audio_embeddings_path, video_embeddings_path, dest_path)
