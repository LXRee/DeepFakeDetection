import os

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data_preparation.embeddings.classes.Mask_RCNN import Mask_RCNN
from data_preparation.utils.make_chunks import list_chunks

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Paths
ABSOLUTE_PATH = "/home/matteo/Projects/Github/DeepFakeDetection/"

RELATIVE_CHECKPOINT_PATH = "data_preparation/embeddings/checkpoints/ASRCNN_27000.pth"
RELATIVE_INPUT_PATH = "dataset/deepfake-detection-challenge/train_sample_extracted_audio"
RELATIVE_OUTPUT_PATH = "data_preparation/embeddings/saved_embeddings"

OUTPUT_FILE_NAME = "test_audio_embeddings.csv"

CHECKPOINT_PATH = os.path.join(ABSOLUTE_PATH, RELATIVE_CHECKPOINT_PATH)
INPUT_PATH = os.path.join(ABSOLUTE_PATH, RELATIVE_INPUT_PATH)
OUTPUT_PATH = os.path.join(ABSOLUTE_PATH, RELATIVE_OUTPUT_PATH)

# Parameters
BATCH_DIM = 64


def load_weights(model, multi_gpu=False):
    if torch.cuda.is_available():
        checkpoint = torch.load(CHECKPOINT_PATH)
    else:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device(DEVICE))
    if multi_gpu:
        model.module.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint['model'], strict=False)
    del checkpoint
    torch.cuda.empty_cache()
    return model


def load_audio_mfcc(audio_path):
    wave, sr = librosa.load(audio_path, mono=True)
    mfcc = librosa.feature.mfcc(wave, sr)
    return mfcc[:, :400] if mfcc.shape[1] > 400 else np.pad(mfcc,
                                                            ((0, 0),
                                                             (0, 400 - len(mfcc[0]))),
                                                            mode='constant',
                                                            constant_values=0)


def predict_on_audio(batch_audio_path, batch_n, audio_model):
    mfccs = []
    for audio_path in tqdm(batch_audio_path, desc="Processing batch {}".format(batch_n), position=0, leave=True):
        try:
            mfcc = load_audio_mfcc(audio_path)
            mfcc = torch.tensor(mfcc, device=DEVICE).float()
            mfcc = torch.unsqueeze(mfcc, dim=0)
            mfccs.append(mfcc)
        except Exception as e:
            print("Prediction error on audio %s: %s" % (audio_path, str(e)))

    outputs = audio_model(torch.cat(mfccs, dim=0))
    return outputs.detach().cpu().numpy()


if __name__ == '__main__':
    audio_paths = [os.path.join(INPUT_PATH, file_name) for file_name in os.listdir(INPUT_PATH)]

    # Prepare batches for inference - to speedup training
    batch_audio_paths = list(list_chunks(audio_paths, BATCH_DIM))
    print("{} batches of {} length".format(len(batch_audio_paths), BATCH_DIM))

    # Prepare model
    model = load_weights(Mask_RCNN()).to(DEVICE)
    model.eval()

    # Create dataframe to store audio logits
    df = pd.DataFrame(columns=['filename', 'audio_embedding'])
    i = 0
    for j, batch in enumerate(batch_audio_paths):
        output_logits = predict_on_audio(batch, j, model)
        for filename, output_logit in zip(batch, output_logits):
            row = [os.path.basename(filename).split('.')[0], output_logit]
            df.loc[i] = row
            i += 1

    # Save the embedding to file
    df.to_pickle(os.path.join(OUTPUT_PATH, OUTPUT_FILE_NAME))
