# -*- coding: utf-8 -*-

import torch
from torch import nn


class Network(nn.Module):

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
        x = self.out(x)

        return x, rnn_state
