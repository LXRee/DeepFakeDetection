import os
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from source.data_preparation.helper.make_chunks import list_chunks
from tqdm import tqdm

# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print(f'Running on device: {device}')


def load_audio_mfcc(audio_path):
    wave, sr = librosa.load(audio_path, mono=True)
    mfcc = librosa.feature.mfcc(wave, sr)
    if mfcc.shape[1] > 400:
        mfcc = mfcc[:, :400]
    else:
        mfcc = np.pad(mfcc, ((0, 0), (0, 400 - len(mfcc[0]))), mode='constant', constant_values=0)
    return mfcc


class MASRCNN_activate(nn.Module):
    def __init__(self, max_sent_len, embedding_dim, num_conv_blocks, init_neurons, num_classes=2):
        super(MASRCNN_activate, self).__init__()
        num_dense_neurons = 50
        convnet_3 = []
        convnet_5 = []
        convnet_7 = []
        for ly in range(0, num_conv_blocks):
            if ly == 0:
                convnet_3.append(nn.Conv1d(in_channels=embedding_dim, out_channels=init_neurons, kernel_size=3))
                convnet_3.append(nn.LeakyReLU(0.2))
                convnet_5.append(nn.Conv1d(in_channels=embedding_dim, out_channels=init_neurons, kernel_size=5))
                convnet_5.append(nn.LeakyReLU(0.2))
                convnet_7.append(nn.Conv1d(in_channels=embedding_dim, out_channels=init_neurons, kernel_size=7))
                convnet_7.append(nn.LeakyReLU(0.2))
            elif ly == 1:
                convnet_3.append(
                    nn.Conv1d(in_channels=init_neurons, out_channels=init_neurons * (ly * 2), kernel_size=3))
                convnet_3.append(nn.LeakyReLU(0.2))
                convnet_5.append(
                    nn.Conv1d(in_channels=init_neurons, out_channels=init_neurons * (ly * 2), kernel_size=5))
                convnet_5.append(nn.LeakyReLU(0.2))
                convnet_7.append(
                    nn.Conv1d(in_channels=init_neurons, out_channels=init_neurons * (ly * 2), kernel_size=7))
                convnet_7.append(nn.LeakyReLU(0.2))
            else:
                convnet_3.append(
                    nn.Conv1d(in_channels=init_neurons * ((ly - 1) * 2), out_channels=init_neurons * (ly * 2),
                              kernel_size=3))
                convnet_3.append(nn.LeakyReLU(0.2))
                convnet_5.append(
                    nn.Conv1d(in_channels=init_neurons * ((ly - 1) * 2), out_channels=init_neurons * (ly * 2),
                              kernel_size=5))
                convnet_5.append(nn.LeakyReLU(0.2))
                convnet_7.append(
                    nn.Conv1d(in_channels=init_neurons * ((ly - 1) * 2), out_channels=init_neurons * (ly * 2),
                              kernel_size=7))
                convnet_7.append(nn.LeakyReLU(0.2))
        self.conv_blocks_3 = nn.Sequential(*convnet_3)
        self.conv_blocks_5 = nn.Sequential(*convnet_5)
        self.conv_blocks_7 = nn.Sequential(*convnet_7)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.dense = nn.Sequential(nn.Linear(448 * 3, num_dense_neurons),
                                   nn.BatchNorm1d(num_dense_neurons),
                                   nn.LeakyReLU(0.2)
                                   )
        self.fc = nn.Linear(50, num_classes)

    def forward(self, x):
        x_3 = self.conv_blocks_3(x)
        x_5 = self.conv_blocks_5(x)
        x_7 = self.conv_blocks_7(x)
        x_3 = self.maxpool(x_3)
        x_5 = self.maxpool(x_5)
        x_7 = self.maxpool(x_7)
        x = torch.cat([x_3, x_5, x_7], 2)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        x = F.dropout(x, p=0.5, training=self.training)
        # x = self.fc(x)
        # skip last fc layer in order to retrieve logits
        return x


def load_weights(model, checkpoint_path, multi_gpu=False):
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    if multi_gpu:
        model.module.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint['model'], strict=False)
    del checkpoint
    torch.cuda.empty_cache()
    return model


def prepare_model(ckp_path, device):
    # checkpoint_path = "/kaggle/input/audio-model/ASRCNN_27000.pth"
    audio_model = MASRCNN_activate(max_sent_len=400, embedding_dim=20, num_conv_blocks=8, init_neurons=32)
    audio_model = load_weights(audio_model, ckp_path)
    audio_model = audio_model.to(device)
    audio_model.eval()
    return audio_model


def predict_on_audio(batch_audio_path, audio_model, batch_n):
    """

    :param audio_path:
    :param audio_model:
    :return: logits features from audio
    """
    mfccs = []
    for audio_path in tqdm(batch_audio_path, desc="Processing batch {}".format(j), position=0, leave=True):
        try:
            mfcc = load_audio_mfcc(audio_path)
            mfcc = torch.tensor(mfcc, device=device).float()
            mfcc = torch.unsqueeze(mfcc, dim=0)
            mfccs.append(mfcc)
        except Exception as e:
            print("Prediction error on audio %s: %s" % (audio_path, str(e)))

    outputs = audio_model(torch.cat(mfccs, dim=0))
    # output = torch.softmax(output, dim=1)
    # pred = output.detach().cpu().numpy()[0][1]
    # return pred
    return outputs.detach().cpu().numpy()


if __name__ == '__main__':
    input_dir = 'test_audio'
    output_dir = ''
    audio_paths = []
    for root, dirs, files in os.walk(input_dir):
        # prepare list of paths to audio
        audio_paths.extend([os.path.join(root, file) for file in files])

    BATCH_DIM = 64
    # prepare batches for inference - to speedup training
    batch_audio_paths = list(list_chunks(audio_paths, BATCH_DIM))
    print("{} batches of {} length".format(len(batch_audio_paths), BATCH_DIM))
    # prepare model
    checkpoint_path = 'source/data_preparation/checkpoints/ASRCNN_27000.pth'

    audio_model = prepare_model(checkpoint_path, device)

    # create dataframe to store audio logits
    df = pd.DataFrame(columns=['filename', 'audio_embedding'])

    # general counter for dataframe
    i = 0
    for j, batch in enumerate(batch_audio_paths):
        output_logits = predict_on_audio(batch, audio_model, j)
        for filename, output_logit in zip(batch, output_logits):
            row = [
                os.path.basename(filename).split('.')[0],
                output_logit
            ]
            df.loc[i] = row
            i += 1

    df.to_pickle('test_audio_embeddings.csv')
