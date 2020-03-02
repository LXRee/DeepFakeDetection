import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Running on device: {DEVICE}')

# Parameters
MAX_SENT_LEN = 400
EMBEDDING_DIM = 20
NUM_CONV_BLOCKS = 8
INIT_NEURONS = 32
NUM_CLASSES = 2


class Mask_RCNN(nn.Module):
    def __init__(self):
        super(Mask_RCNN, self).__init__()
        num_dense_neurons = 50
        convnet_3 = []
        convnet_5 = []
        convnet_7 = []
        for ly in range(0, NUM_CONV_BLOCKS):
            if ly == 0:
                convnet_3.append(nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=INIT_NEURONS, kernel_size=3))
                convnet_3.append(nn.LeakyReLU(0.2))
                convnet_5.append(nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=INIT_NEURONS, kernel_size=5))
                convnet_5.append(nn.LeakyReLU(0.2))
                convnet_7.append(nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=INIT_NEURONS, kernel_size=7))
                convnet_7.append(nn.LeakyReLU(0.2))
            elif ly == 1:
                convnet_3.append(nn.Conv1d(in_channels=INIT_NEURONS,
                                           out_channels=INIT_NEURONS * (ly * 2),
                                           kernel_size=3))
                convnet_3.append(nn.LeakyReLU(0.2))
                convnet_5.append(nn.Conv1d(in_channels=INIT_NEURONS,
                                           out_channels=INIT_NEURONS * (ly * 2),
                                           kernel_size=5))
                convnet_5.append(nn.LeakyReLU(0.2))
                convnet_7.append(nn.Conv1d(in_channels=INIT_NEURONS,
                                           out_channels=INIT_NEURONS * (ly * 2),
                                           kernel_size=7))
                convnet_7.append(nn.LeakyReLU(0.2))
            else:
                convnet_3.append(nn.Conv1d(in_channels=INIT_NEURONS * ((ly - 1) * 2),
                                           out_channels=INIT_NEURONS * (ly * 2),
                                           kernel_size=3))
                convnet_3.append(nn.LeakyReLU(0.2))
                convnet_5.append(nn.Conv1d(in_channels=INIT_NEURONS * ((ly - 1) * 2),
                                           out_channels=INIT_NEURONS * (ly * 2),
                                           kernel_size=5))
                convnet_5.append(nn.LeakyReLU(0.2))
                convnet_7.append(nn.Conv1d(in_channels=INIT_NEURONS * ((ly - 1) * 2),
                                           out_channels=INIT_NEURONS * (ly * 2),
                                           kernel_size=7))
                convnet_7.append(nn.LeakyReLU(0.2))

        self.conv_blocks_3 = nn.Sequential(*convnet_3)
        self.conv_blocks_5 = nn.Sequential(*convnet_5)
        self.conv_blocks_7 = nn.Sequential(*convnet_7)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.dense = nn.Sequential(nn.Linear(448 * 3, num_dense_neurons),
                                   nn.BatchNorm1d(num_dense_neurons),
                                   nn.LeakyReLU(0.2))

        self.fc = nn.Linear(50, NUM_CLASSES)

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
        return x
