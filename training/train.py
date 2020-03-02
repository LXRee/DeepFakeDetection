import argparse
import json
import os
from itertools import product
from typing import Dict

import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

from training.dataset import EmbeddingsDataset, ToTensor, RandomCrop
from training.model import Model

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('Running on device: {}'.format(DEVICE))

# ------------------
# --- PARAMETERS ---
# ------------------

parser = argparse.ArgumentParser(description='Train the deepfake network.')

# Dataset
parser.add_argument('--datasetpath',
                    type=str,
                    default='dataset/train_audio_video_embeddings.csv',
                    help='Path of the train csv folder')
parser.add_argument('--testdatasetpath',
                    type=str,
                    default='dataset/test_audio_video_embeddings.csv',
                    help='Path of the test csv folder')
parser.add_argument('--crop_len',
                    type=int,
                    default=10,
                    help='Number of frames features to be randomly cropped from video')

# Network structure
parser.add_argument('--video_embedding_dim', type=int, default=512, help='Dimension of features vector of video')
parser.add_argument('--audio_embedding_dim', type=int, default=50, help='Dimension of features vector of audio')
parser.add_argument('--network', type=str, default='transformer', help='Network to use')

# Model hyperparameters
parser.add_argument('--fc_dim', type=int, default=512, help='Dimension of last FC layer that collects video and audio')
parser.add_argument('--dropout_prob', type=float, default=0.3, help='Dropout probability')

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
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--num_epochs', type=int, default=100000, help='Number of training epochs')
parser.add_argument('--patience', type=int, default=15, help='Patience to use in EarlyStopping')

# Save
parser.add_argument('--model_dir', type=str, default='exp12', help='Where to load from models and params')

args = parser.parse_args()
NETWORK = args.network

TRAIN_DATASET_PATH = args.datasetpath
TEST_DATASET_PATH = args.testdatasetpath
CROP_LEN = args.crop_len

VIDEO_EMBEDDING_DIM = args.video_embedding_dim
AUDIO_EMBEDDING_DIM = args.audio_embedding_dim
FC_DIM = args.fc_dim
DROPOUT_PROB = args.dropout_prob
# LSTM
HIDDEN_UNITS = args.hidden_units
LAYERS_NUM = args.layers_num
# Tranformer
N_HEAD = args.n_head
DIM_FEEDFORWARD = args.dim_feedforward
ENC_LAYERS = args.enc_layers
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
crop_len = [5, 10, 15, 25]
learning_rate = [1e-03]
dropout_prob = [0.3]
batch_size = [256]
fc_dim = [256, 512, 784]

if NETWORK == 'LSTM':
    hidden_units = [128, 256, 512]
    layers_num = [2, 3, 5]
elif NETWORK == 'transformer':
    n_head = [8]
    dim_feedforward = [2048]
    enc_layers = [6]
else:
    raise ValueError('{} network has not been implemented yet. Please choose between "LSTM" and "transformer"')


def clean_folder(folder):
    """
    Cleans all checkpoints that are sub-optimal and leave the best.
    :param folder: folder path
    :return: None
    """
    min_loss = 100
    for file in os.listdir(folder):
        if 'checkpoint' in file:
            filename = file.split('.')[0] + '.' + file.split('.')[1]
            loss_value = filename.split('_')[1]
            loss_value = float(loss_value)
            if loss_value < min_loss:
                min_loss = loss_value
            else:
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
        FC_DIM = parameters['fc_dim']

        # overwrite hyper parameters args to save them in the right way
        args.crop_len = CROP_LEN
        args.learning_rate = LEARNING_RATE
        args.dropout_prob = DROPOUT_PROB
        args.batchsize = BATCH_SIZE
        args.fc_dim = FC_DIM
        # Define network params
        net_params = {
            'fc_dim': FC_DIM,
            'dropout_prob': DROPOUT_PROB,
            'video_embedding_dim': VIDEO_EMBEDDING_DIM,
            'audio_embedding_dim': AUDIO_EMBEDDING_DIM
        }

        if NETWORK == 'LSTM':
            HIDDEN_UNITS = parameters['hidden_units']
            LAYERS_NUM = parameters['layers_num']
            RUN_PATH = os.path.join('source', 'training', 'experiments', NETWORK,
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

            # add LST params
            net_params['hidden_units'] = HIDDEN_UNITS
            net_params['layers_num'] = LAYERS_NUM

        elif NETWORK == 'transformer':
            N_HEAD = parameters['n_head']
            DIM_FEEDFORWARD = parameters['dim_feedforward']
            ENC_LAYERS = parameters['enc_layers']
            RUN_PATH = os.path.join('source', 'training', 'experiments', NETWORK,
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
            # initialize CUDA state at every iteration
            torch.cuda.empty_cache()
            torch.cuda.init()

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

            dataset_len = len(dataset)
            partitions = round(dataset_len * (1 - VAL_SIZE)), round(dataset_len * VAL_SIZE)
            train_set, val_set = random_split(dataset, [*partitions])
            train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
            val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)

            model = Model(
                NETWORK,
                net_params,
                dataset.get_pos_weight,
                OPTIMIZER,
                LOSS_TYPE,
                LEARNING_RATE,
            )
            model.fit(
                epochs=NUM_EPOCHS,
                train_set=train_loader,
                val_set=val_loader,
                patience=PATIENCE,
                run_name=RUN_PATH
            )

            # clean folder from useless checkpoints
            clean_folder(RUN_PATH)
        except FileExistsError as e:
            print("Folder {} already trained. Jumping to next hyper parameters.".format(RUN_PATH))


def __train__():
    dataset = EmbeddingsDataset(csv_path=TRAIN_DATASET_PATH)

    # select grid search based on network type
    if NETWORK == 'LSTM':
        for CROP_LEN, HIDDEN_UNITS, LAYERS_NUM, LEARNING_RATE, DROPOUT_PROB, BATCH_SIZE, FC_DIM in product(
                crop_len, hidden_units, layers_num, learning_rate, dropout_prob, batch_size, fc_dim
        ):
            parameters = {
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
        for CROP_LEN, N_HEAD, DIM_FEEDFORWARD, ENC_LAYERS, LEARNING_RATE, DROPOUT_PROB, BATCH_SIZE, FC_DIM in product(
                crop_len, n_head, dim_feedforward, enc_layers, learning_rate, dropout_prob, batch_size, fc_dim
        ):
            parameters = {
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


def __evaluate__():
    # Load training parameters
    model_dir = os.path.join('training', 'experiments', args.model_dir)
    print('Loading model from: %s' % model_dir)
    training_args = json.load(open(os.path.join(model_dir, 'training_args.json')))

    trans = transforms.Compose([RandomCrop(CROP_LEN),
                                # LabelOneHot(),
                                ToTensor()])

    test_set = EmbeddingsDataset(csv_path=TEST_DATASET_PATH, transform=trans)
    test_loader = DataLoader(test_set, batch_size=len(test_set), num_workers=0, pin_memory=True)
    # Restore hyper parameters based on network type
    network_type = training_args['network']
    net_params = {
        'dropout_prob': training_args['dropout_prob'],
        'fc_dim': training_args['fc_dim'],
        'video_embedding_dim': training_args['video_embedding_dim'],
        'audio_embedding_dim': training_args['audio_embedding_dim']
    }
    if network_type == 'LSTM':
        net_params['hidden_units'] = training_args['hidden_units']
        net_params['layers_num'] = training_args['layers_num']
    elif network_type == 'transformer':
        net_params['n_head'] = training_args['n_head']
        net_params['dim_feedforward'] = training_args['dim_feedforward']
        net_params['enc_layers'] = training_args['enc_layers']
    else:
        raise ValueError("Bad network type. Please be sure to restore models from 'LSTM' or 'transfomer' networks")

    model = Model(network_type,
                  net_params,
                  test_set.get_pos_weight,
                  training_args['optimizer'],
                  training_args['loss_type'],
                  training_args['learning_rate'])

    # Load network trained parameters and evaluate
    model.net.load_state_dict(torch.load(os.path.join(model_dir, 'checkpoint_0.132483.pt'))['state_dict'])
    model.evaluate(test_loader)


if __name__ == '__main__':
    __train__()
    # __evaluate__()
