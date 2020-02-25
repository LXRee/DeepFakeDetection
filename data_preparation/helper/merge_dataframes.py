import pandas as pd
import os


def merge_dataframes(source, dest):
    """
    Takes all dataframes in source folder and merge into a single Dataframe saved in dest folder
    :param source: source folder of the embeddings
    :param dest: destination folder for the embeddings
    :return: None
    """
    df = pd.DataFrame(columns=['filename', 'embedding', 'label'])
    for root, dirs, files in os.walk(source):
        for file in files:
            df_part = pd.read_csv(os.path.join(root, file))
            df = pd.concat([df, df_part], axis=1)
    df.to_csv(os.path.join(dest, 'merged.csv'), index=False)
