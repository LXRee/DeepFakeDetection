# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer


class LSTMSiameseNetwork(nn.Module):
    def __init__(self,
                 hidden_units,
                 layers_num,
                 fc_dim,
                 video_embedding_dim=512,
                 audio_embedding_dim=256,
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
        self.dropout = nn.Dropout(dropout_prob)

        # Layer that receives the two branches.
        # Keep it little in order to move a better representation search of rnn network
        self.concat_branches = nn.Linear(fc_dim * 2, 64)
        self.dropout_last = nn.Dropout(0.3)
        self.out = nn.Linear(64, 2)

    def forward(self, inputs, state=None):
        # LSTM for video information
        video_embedding_0, audio_embedding_0, video_embedding_1, audio_embedding_1 = inputs
        x_0 = self.branch(video_embedding_0, audio_embedding_0)
        x_1 = self.branch(video_embedding_1, audio_embedding_1)
        x = self.concat_branches(torch.cat([x_0, x_1], dim=1))
        x = self.dropout_last(x)
        x = self.out(F.relu(x))
        return x

    def branch(self, video_embedding, audio_embedding, state=None):
        x, _ = self.rnn(video_embedding, state)
        # Concatenating audio information
        # We want to consider only the last time step for video since we want to understand what the LSTM has seen
        x = self.fc(torch.cat([x[:, -1, :], audio_embedding], dim=1))
        x = self.dropout(x)
        x = F.relu(x)
        return x


class TransformerSiameseNetwork(nn.Module):
    def __init__(self,
                 n_head,
                 dim_feedforward,
                 enc_layers,
                 dropout_prob,
                 fc_dim,
                 video_embedding_dim,
                 audio_embedding_dim=256, ):
        # Call the parent init function (required!)
        super().__init__()

        # Define TransformerEncoder layer instance:
        encoder_layer = TransformerEncoderLayer(video_embedding_dim, n_head, dim_feedforward, dropout_prob)

        # Define Transformer layer
        self.trans = TransformerEncoder(encoder_layer, enc_layers)

        # FC layer to let the network decide how much audio will be considered to decide the result
        self.fc = nn.Linear(video_embedding_dim + audio_embedding_dim, fc_dim)

        # dropout layer after linear layer
        self.dropout = nn.Dropout(dropout_prob)

        # Layer that receives the two branches.
        # Keep it little in order to move a better representation search of rnn network
        self.concat_branches = nn.Linear(fc_dim * 2, 64)
        self.dropout_last = nn.Dropout(0.3)
        # Define output layer
        self.out = nn.Linear(64, 2)

    def branch(self, video_embedding, audio_embedding, state=None):
        # LSTM for video information
        x = self.trans(video_embedding, state)

        # Concatenating audio information
        # We want to consider only the last time step for video since we want to understand what the LSTM has seen
        x = self.fc(torch.cat([x[:, -1, :], audio_embedding], dim=1))

        x = self.dropout(x)

        return x

    def forward(self, inputs, state=None):
        # Transformer for video information
        video_embedding_0, audio_embedding_0, video_embedding_1, audio_embedding_1 = inputs
        x_0 = self.branch(video_embedding_0, audio_embedding_0)
        x_1 = self.branch(video_embedding_1, audio_embedding_1)
        x = self.concat_branches(torch.cat([x_0, x_1], dim=1))
        x = self.dropout_last(x)
        x = self.out(F.relu(x))
        return x
