import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict


def read_csv(submission_csv, dataset_csv) -> float:
    """
    Read csv and return dict containing filename:prob
    :param csv_path:
    :return:
    """
    submission_df = pd.read_csv(submission_csv)
    target_df = pd.read_pickle(dataset_csv)
    submission = {submission_df['filename'].loc[i]: submission_df['label'].loc[i] for i in tqdm(range(submission_df['filename'].shape[0]), desc='Reading submission')}
    target = {target_df['filename'].loc[i]: target_df['label'].loc[i] for i in tqdm(range(target_df['filename'].shape[0]) , desc='Reading target')}
    target_probs = []
    submission_probs = []
    for k, v in tqdm(target.items(), desc='Creating probs vector'):
        target_probs.append(target[k])
        submission_probs.append(submission[k])
    target_probs = torch.tensor(target_probs).float()
    submission_probs = torch.tensor(submission_probs)
    logloss = torch.nn.functional.binary_cross_entropy(submission_probs, target_probs).detach().cpu().numpy()
    return logloss


if __name__ == '__main__':
    submission_csv = 'submission.csv'
    dataset_csv = 'dataset/new_test_audio_video_embeddings.csv'
    print(read_csv(submission_csv, dataset_csv))


