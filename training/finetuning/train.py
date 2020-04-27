import argparse
import json
import os
import random
from itertools import product
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from training.finetuning.dataset import EmbeddingsDataset, ToTensor, RandomCrop
from training.finetuning.model import Model

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('Running on device: {}'.format(DEVICE))

# ------------------
# --- PARAMETERS ---
# ------------------

parser = argparse.ArgumentParser(description='Train the deepfake network.')

# Dataset
parser.add_argument('--dataset_reals_path',
                    type=str,
                    default='dataset/real_train.csv',
                    help='Path of the true videos csv csv file')
parser.add_argument('--dataset_fakes_path',
                    type=str,
                    default='dataset/fake_train.csv',
                    help='Path of the fake videos csv file')
parser.add_argument('--checkpoint_path',
                    type=str,
                    default='source/training/siamese/experiments/transformer/crop6_head16_dimF2048_encL6_lr0.0001_fc1024_batch512_drop0.3/checkpoint_0.15873556_ep49.pt',
                    help='Path of the pre-trained model')
parser.add_argument('--crop_len',
                    type=int,
                    default=40,
                    help='Number of frames features to be randomly cropped from video')

# Network structure
parser.add_argument('--video_embedding_dim', type=int, default=512, help='Dimension of features vector of video')
parser.add_argument('--audio_embedding_dim', type=int, default=256, help='Dimension of features vector of audio')
parser.add_argument('--network', type=str, default='LSTM', help='Network to use')

# Model hyperparameters
parser.add_argument('--fc_dim', type=int, default=512, help='Dimension of last FC layer that collects video and audio')
parser.add_argument('--dropout_prob', type=float, default=0.5, help='Dropout probability')

# LSTM
parser.add_argument('--hidden_units', type=int, default=256, help='Number of RNN hidden units')
parser.add_argument('--layers_num', type=int, default=3, help='Number of RNN stacked layers')

# TRANSFORMER
parser.add_argument('--n_head', type=int, default=8, help='The number of heads in the multiheadattention models')
parser.add_argument('--dim_feedforward', type=int, default=2048, help='The dimension of the feedforward network model')
parser.add_argument('--enc_layers', type=int, default=6, help='The number of sub-encoder-layers in the encoder')

# Training parameters
parser.add_argument('--batchsize', type=int, default=128, help='Training batch size')
parser.add_argument('--optimizer', type=str, default='adam', help='Type of optimizer')
parser.add_argument('--loss_type', type=str, default='BCE', help='Loss type')
parser.add_argument('--val_size', type=float, default=.3, help='Dimension of validation')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--num_epochs', type=int, default=100000, help='Number of training epochs')
parser.add_argument('--patience', type=int, default=2, help='Patience to use in EarlyStopping')

# Save
parser.add_argument('--model_dir', type=str, default='crop40_hid512_ln2_lr0.001_fc512_batch256_drop0.5',
                    help='Where to load from models and params')

args = parser.parse_args()
FOLDS = 3

REAL_DATASET_PATH = args.dataset_reals_path
FAKE_DATASET_PATH = args.dataset_fakes_path
CHECKPOINT_PATH = args.checkpoint_path
# Retrieve network args from checkpoint
network_args = json.load(open(os.path.join(os.path.dirname(CHECKPOINT_PATH), 'training_args.json'), 'r'))
# Network type
NETWORK = network_args['network']
VIDEO_EMBEDDING_DIM = network_args['video_embedding_dim']
AUDIO_EMBEDDING_DIM = network_args['audio_embedding_dim']
FC_DIM = network_args['fc_dim']
# LSTM
HIDDEN_UNITS = network_args['hidden_units']
LAYERS_NUM = network_args['layers_num']
# Tranformer
N_HEAD = network_args['n_head']
DIM_FEEDFORWARD = network_args['dim_feedforward']
ENC_LAYERS = network_args['enc_layers']


# Training args
CROP_LEN = args.crop_len
DROPOUT_PROB = args.dropout_prob
# Training
BATCH_SIZE = args.batchsize
LEARNING_RATE = args.learning_rate
NUM_EPOCHS = args.num_epochs

OPTIMIZER = args.optimizer
LOSS_TYPE = args.loss_type
VAL_SIZE = args.val_size
PATIENCE = args.patience

# Parameters for grid search.
# These hyperparameters overwrite the one parsed by argparse, so you should change
# their values here.
crop_len = [6]
learning_rate = [1e-3]
dropout_prob = [0.3]
batch_size = [512]


def clean_folder(folder, loss, delta=0.02):
    """
    Cleans all checkpoints that are distant > delta from average loss
    :param folder: folder path
    :return: None
    """
    for file in os.listdir(folder):
        if 'checkpoint' in file:
            filename = file.split('.')[0] + '.' + file.split('.')[1]
            loss_value = filename.split('_')[1]
            loss_value = float(loss_value)
            if not loss - delta < loss_value < loss + delta:
                os.remove(os.path.join(folder, file))


def do_the_job(parameters: Dict, dataset):
    """
    Actual training. This method is useful to keep compatibility between network architectures
    :param parameters: dict with parameters
    :return:
    """
    CROP_LEN = parameters['crop_len']
    LEARNING_RATE = parameters['learning_rate']
    DROPOUT_PROB = parameters['dropout_prob']
    BATCH_SIZE = parameters['batch_size']

    # overwrite hyper parameters args to save them in the right way
    args.crop_len = CROP_LEN
    args.learning_rate = LEARNING_RATE
    args.dropout_prob = DROPOUT_PROB
    args.batchsize = BATCH_SIZE
    args.fc_dim = FC_DIM
    # Define network params
    net_params = {
        'checkpoint_path': CHECKPOINT_PATH,
        'fc_dim': FC_DIM,
        'dropout_prob': DROPOUT_PROB,
        'video_embedding_dim': VIDEO_EMBEDDING_DIM,
        'audio_embedding_dim': AUDIO_EMBEDDING_DIM
    }

    if NETWORK == 'LSTM':
        RUN_PATH = os.path.join('source', 'training', 'finetuning', 'experiments', NETWORK,
                                'crop{crop}_hid{hid}_ln{ln}_lr{lr}_fc{fc}_batch{b}_drop{d}'
                                .format(crop=CROP_LEN,
                                        hid=HIDDEN_UNITS,
                                        ln=LAYERS_NUM,
                                        lr=LEARNING_RATE,
                                        d=DROPOUT_PROB,
                                        b=BATCH_SIZE,
                                        fc=FC_DIM))
        # overwrite parameters that has been choosen from gridsearch
        args.hidden_units = HIDDEN_UNITS
        args.layers_num = LAYERS_NUM

        # add LSTM params
        net_params['hidden_units'] = HIDDEN_UNITS
        net_params['layers_num'] = LAYERS_NUM

    elif NETWORK == 'transformer':
        RUN_PATH = os.path.join('source', 'training', 'finetuning', 'experiments', NETWORK,
                                'crop{crop}_head{head}_dimF{dimF}_encL{encL}_lr{lr}_fc{fc}_batch{b}_drop{d}'
                                .format(crop=CROP_LEN,
                                        head=N_HEAD,
                                        dimF=DIM_FEEDFORWARD,
                                        encL=ENC_LAYERS,
                                        lr=LEARNING_RATE,
                                        d=DROPOUT_PROB,
                                        b=BATCH_SIZE,
                                        fc=FC_DIM))
        # overwrite parameters that has been choosen from gridsearch
        args.n_head = N_HEAD
        args.dim_feedforward = DIM_FEEDFORWARD
        args.enc_layers = ENC_LAYERS

        # Add transformer params
        net_params['n_head'] = N_HEAD
        net_params['dim_feedforward'] = DIM_FEEDFORWARD
        net_params['enc_layers'] = ENC_LAYERS
    else:
        raise ValueError('Bad network type. Please choose between "LSTM" and "transformer"')

    print("Now training at: \n{}".format(RUN_PATH))

    try:
        os.makedirs(RUN_PATH, exist_ok=False)

        trans = transforms.Compose([
            RandomCrop(CROP_LEN),
            # LabelOneHot(),
            ToTensor()
        ])

        dataset.transform = trans

        # Save training parameters
        # TODO: it is saving ALL parameters - from all networks. It should save only parameters for the right one
        with open(os.path.join(RUN_PATH, 'training_args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

        # Prepare information about dataset and partitions
        dataset_len = len(dataset)
        # partitions = round(dataset_len * (1 - VAL_SIZE)), round(dataset_len * VAL_SIZE)
        # Prepare for kFold
        indexes = list(range(dataset_len))
        random.shuffle(indexes)
        random.shuffle(indexes)
        indexes_per_fold = round(dataset_len * VAL_SIZE)

        # define loss array
        losses = np.zeros(FOLDS, 'float32')
        # This indexes selection has been tested and the subsets are without overlapping indexes
        for i in range(FOLDS):
            print("Training fold {}/{}".format(i + 1, FOLDS))
            val_set = Subset(dataset, indexes[i * indexes_per_fold: (i + 1) * indexes_per_fold])
            train_set = Subset(dataset, indexes[0: i * indexes_per_fold] + indexes[(i + 1) * indexes_per_fold:])

            train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
            val_loader = DataLoader(val_set, batch_size=round(BATCH_SIZE * VAL_SIZE), num_workers=0, pin_memory=True)
            model = Model(
                NETWORK,
                net_params,
                OPTIMIZER,
                LOSS_TYPE,
                LEARNING_RATE,
            )
            losses[i] = model.fit(
                epochs=NUM_EPOCHS,
                train_set=train_loader,
                val_set=val_loader,
                patience=PATIENCE,
                run_name=RUN_PATH
            )
        loss = losses.mean()
        with open(os.path.join(RUN_PATH, 'average_loss-{}-.txt').format(loss), 'w') as f:
            f.write('Average loss over {} folds: {:.4f}'.format(FOLDS, loss))

        print(
            "Average loss over these parameters: {}\nPlease choose the checkpoint that is nearer to this value.".format(
                loss))
        # clean folder from useless checkpoints
        clean_folder(RUN_PATH, loss)

    except FileExistsError:
        print("Folder {} already trained. Jumping to next hyper parameters.".format(RUN_PATH))


def __train__():
    dataset = EmbeddingsDataset(real_csv_path=REAL_DATASET_PATH, fake_csv_path=FAKE_DATASET_PATH)
    # select grid search based on network type

    if NETWORK == 'LSTM':
        for CROP_LEN, LEARNING_RATE, DROPOUT_PROB, BATCH_SIZE in product(
                crop_len, learning_rate, dropout_prob, batch_size
        ):
            parameters = {
                'checkpoint_path': CHECKPOINT_PATH,
                'crop_len': CROP_LEN,
                'learning_rate': LEARNING_RATE,
                'dropout_prob': DROPOUT_PROB,
                'batch_size': BATCH_SIZE,
                'fc_dim': FC_DIM,
                'hidden_units': HIDDEN_UNITS,
                'layers_num': LAYERS_NUM
            }
            do_the_job(parameters, dataset)

    elif NETWORK == 'transformer':
        for CROP_LEN, LEARNING_RATE, DROPOUT_PROB, BATCH_SIZE in product(
                crop_len, learning_rate, dropout_prob, batch_size
        ):
            parameters = {
                'checkpoint_path': CHECKPOINT_PATH,
                'crop_len': CROP_LEN,
                'learning_rate': LEARNING_RATE,
                'dropout_prob': DROPOUT_PROB,
                'batch_size': BATCH_SIZE,
                'fc_dim': FC_DIM,
                'n_head': N_HEAD,
                'dim_feedforward': DIM_FEEDFORWARD,
                'enc_layers': ENC_LAYERS
            }
            do_the_job(parameters, dataset)
    else:
        raise ValueError('Bad network type for {}. Please choose between "LSTM" or "transformer"'.format(NETWORK))


if __name__ == '__main__':
    __train__()
    # __evaluate__()
