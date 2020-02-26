import pandas as pd
import os
from tqdm import tqdm


def merge_dataframes(source, dest):
    """
    Takes all dataframes in source folder and merge into a single Dataframe saved in dest folder
    :param source: source folder of the embeddings
    :param dest: destination folder for the embeddings
    :return: None
    """
    df = pd.DataFrame(columns=['filename', 'embedding', 'label'])
    for root, dirs, files in os.walk(source):
        for file in tqdm(files, desc="Reading source"):
            df_part = pd.read_pickle(os.path.join(root, file))
            df = pd.concat([df, df_part])

    initial_df_dim = df['label'].shape[0]
    # sanity checks
    deletion_indexes = []
    seq_threshold = 3
    no_embeddings = 0
    small_seq = 0
    for e in tqdm(range(df['embedding'].shape[0]), desc='Sanity checks'):
        # check if every embedding has 2 dimensions
        if len(df['embedding'].loc[e].shape) is not 2:
            deletion_indexes.append(e)
            no_embeddings += 1
        # check if the sequence dimension is too small (above 3 -> 4*6 (window) = 24 frames, at least 1 sec video)
        if df['embedding'].loc[e].shape[0] < seq_threshold:
            deletion_indexes.append(e)
            small_seq += 1
    print("Found {} None embeddings and {} too small sequences".format(no_embeddings, small_seq))
    deletion_indexes.sort()
    for index in tqdm(deletion_indexes, desc="Deletion of bad rows"):
        df = df.drop(df.index[index])
    df = df.reset_index(drop=True)

    final_df_dim = df['label'].shape[0]
    print("Initial df dimension: {}\nFinal df dimension after sanity checks: {}".format(initial_df_dim, final_df_dim))
    print('Now saving...')
    df.to_pickle(os.path.join(dest, 'merged.csv'))
    print('Merge saved at file: {}'.format(os.path.join(dest, 'merged.gzip')))


if __name__ == '__main__':
    merge_dataframes('embeddings/partials', 'embeddings')