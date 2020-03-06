# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer


class LSTMNetwork(nn.Module):
    def __init__(self,
                 hidden_units,
                 layers_num,
                 fc_dim,
                 video_embedding_dim=512,
                 audio_embedding_dim=50,
                 dropout_prob=0.):
        # Call the parent init function (required!)
        super().__init__()

        # Define recurrent layer
        self.rnn = nn.LSTM(input_size=video_embedding_dim,
                           hidden_size=hidden_units,
                           num_layers=layers_num,
                           dropout=dropout_prob,
                           batch_first=True)

        # FC layer to let the network decide how much audio will be considered to decide the result
        self.fc = nn.Linear(hidden_units + audio_embedding_dim, fc_dim)

        # dropout layer after linear layer
        # self.dropout = nn.Dropout(dropout_prob)

        # Define output layer
        self.out = nn.Linear(fc_dim, 1)

    def forward(self, inputs, state=None):
        # LSTM for video information
        x, rnn_state = self.rnn(inputs[0], state)

        # Concatenating audio information
        # We want to consider only the last time step for video since we want to understand what the LSTM has seen
        x = self.fc(torch.cat([x[:, -1, :], inputs[1]], dim=1))

        # x = self.dropout(x)

        # Linear layer
        x = self.out(F.relu(x))

        return x


class TransformerNetwork(nn.Module):
    def __init__(self,
                 n_head,
                 dim_feedforward,
                 enc_layers,
                 dropout_prob,
                 fc_dim,
                 video_embedding_dim,
                 audio_embedding_dim=50, ):
        # Call the parent init function (required!)
        super().__init__()

        # Define TransformerEncoder layer instance:
        encoder_layer = TransformerEncoderLayer(video_embedding_dim, n_head, dim_feedforward, dropout_prob)

        # Define Transformer layer
        self.trans = TransformerEncoder(encoder_layer, enc_layers)

        # FC layer to let the network decide how much audio will be considered to decide the result
        self.fc = nn.Linear(video_embedding_dim + audio_embedding_dim, fc_dim)

        # dropout layer after linear layer
        # self.dropout = nn.Dropout(dropout_prob)

        # Define output layer
        self.out = nn.Linear(fc_dim, 1)

    def forward(self, inputs, state=None):
        # LSTM for video information
        x = self.trans(inputs[0], state)

        # Concatenating audio information
        # We want to consider only the last time step for video since we want to understand what the LSTM has seen
        x = self.fc(torch.cat([x[:, -1, :], inputs[1]], dim=1))

        # x = self.dropout(x)

        # Linear layer
        x = self.out(x)

        return x
