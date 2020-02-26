import torch
import os
import argparse
import json
from typing import Union, List, Dict
from torchvision import transforms
from torch.utils.data import random_split, DataLoader

from source.training.dataset import EmbeddingsDataset, ToTensor, RandomCrop, LabelOneHot
from source.training.model import Model

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Running on device: {device}')


##############################
## PARAMETERS
##############################
parser = argparse.ArgumentParser(description='Train the deepfake network.')

# Dataset
parser.add_argument('--datasetpath',    type=str,   default='embeddings/partials/train_embeddings_1.csv', help='Path of the train csv file')
parser.add_argument('--crop_len',       type=int,   default=4,               help='Number of frames features to be randomly cropped from video')

# Network
parser.add_argument('--embedding_dim',   type=int,   default=512,    help='Dimension of features vector of video')
parser.add_argument('--hidden_units',   type=int,   default=64,    help='Number of RNN hidden units')
parser.add_argument('--layers_num',     type=int,   default=1,      help='Number of RNN stacked layers')
parser.add_argument('--dropout_prob',   type=float, default=0.3,    help='Dropout probability')
parser.add_argument('--optimizer',      type=str,   default='adam', help='Type of optimizer')
parser.add_argument('--loss_type',      type=str, default='BCE',    help='Loss type')

# Training
parser.add_argument('--batchsize',      type=int,   default=11,   help='Training batch size')
parser.add_argument('--val_size',      type=float,   default=.3,   help='Dimension of validation')
parser.add_argument('--learning_rate',  type=float,   default=1e-3,   help='Learning rate')
parser.add_argument('--num_epochs',     type=int,   default=100000,    help='Number of training epochs')
parser.add_argument('--patience',     type=int,   default=200,    help='Patience to use in EarlyStopping')

# Save
parser.add_argument('--out_dir',     type=str,   default='exp0',    help='Where to save models and params')

args = parser.parse_args()

DATASET_PATH = args.datasetpath
CROP_LEN = args.crop_len

EMBEDDING_DIM = args.embedding_dim
HIDDEN_UNITS = args.hidden_units
LAYERS_NUM = args.layers_num
DROPOUT_PROB = args.dropout_prob
BATCH_SIZE = args.batchsize
LEARNING_RATE = args.learning_rate
NUM_EPOCHS = args.num_epochs
RUN_PATH = os.path.join('source', 'training', 'experiments', args.out_dir)

OPTIMIZER = args.optimizer
LOSS_TYPE = args.loss_type
VAL_SIZE = args.val_size
PATIENCE = args.patience


def __train__():
    dataset = EmbeddingsDataset(csv_path=DATASET_PATH, shuffle=True)
    net_params = {
        'hidden_units': HIDDEN_UNITS,
        'layers_num': LAYERS_NUM,
        'dropout_prob': DROPOUT_PROB
    }

    os.makedirs(RUN_PATH, exist_ok=True)

    trans = transforms.Compose([
        RandomCrop(CROP_LEN),
        LabelOneHot(),
        ToTensor()
    ])

    dataset.transform = trans

    # Save training parameters
    with open(os.path.join(RUN_PATH, 'training_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    dataset_len = len(dataset)
    partitions = int(dataset_len * (1-VAL_SIZE)), round(dataset_len * VAL_SIZE)
    train_set, val_set = random_split(dataset, [*partitions])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=len(val_set), num_workers=0, pin_memory=True)

    # reset weights to do a trial
    model = Model(
        net_params,
        dataset.get_pos_weight,
        OPTIMIZER,
        LOSS_TYPE,
        LEARNING_RATE,
        EMBEDDING_DIM,
    )
    model.fit(
        epochs=NUM_EPOCHS,
        train_set=train_loader,
        val_set=val_loader,
        patience=PATIENCE,
        run_name=RUN_PATH
    )
    with open(os.path.join(RUN_PATH, 'last_epoch_value.txt'), 'w') as f:
        f.write(str(model.last_epoch))


if __name__ == '__main__':
    __train__()
