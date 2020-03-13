import random
import pandas as pd
from tqdm import tqdm


def df_to_dict(df):
    return {df['filename'].loc[i]: {'video_embedding': df['video_embedding'].loc[i],
                                    'audio_embedding': df['audio_embedding'].loc[i], 'label': df['label'].loc[i]} for i
            in range(df['filename'].shape[0])}


if __name__ == '__main__':
    df_original = pd.read_pickle('dataset/train_audio_video_embeddings.csv')
    df_train = pd.DataFrame(columns=['filename', 'video_embedding', 'audio_embedding', 'label'])
    df_test = pd.DataFrame(columns=['filename', 'video_embedding', 'audio_embedding', 'label'])
    or_keys = list(df_original['filename'])
    random.shuffle(or_keys)
    or_dict = df_to_dict(df_original)

    for i, k in enumerate(or_keys):
        row = [k, or_dict[k]['video_embedding'], or_dict[k]['audio_embedding'], or_dict[k]['label']]

        if i < 4000:
            df_test.loc[df_test['filename'].shape[0]] = row
        else:
            df_train.loc[df_train['filename'].shape[0]] = row

    df_train.to_pickle('dataset/part_train_audio_video_embeddings.csv')
    df_test.to_pickle('dataset/new_test_audio_video_embeddings.csv')

