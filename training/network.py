# -*- coding: utf-8 -*-

from torch import nn
import torch
import torch.nn.functional as F


class Network(nn.Module):

    def __init__(self,
                 hidden_units,
                 layers_num,
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
        self.fc = nn.Linear(hidden_units + audio_embedding_dim, 512)
        # Define output layer
        self.out = nn.Linear(512, 1)

    def forward(self, inputs, state=None):
        # LSTM for video information
        x, rnn_state = self.rnn(inputs[0], state)
        # concatenating audio information
        # we want to consider only the last time step for video since we want to understand what
        # the lstm has seen.
        x = self.fc(torch.cat([x[:, -1, :], inputs[1]], dim=1))
        # Linear layer
        x = self.out(x)
        return x, rnn_state


# def train_batch(net, batch, loss_fn, optimizer, acc_fn):
#     # batch input comes as sparse
#
#     # Get the labels (the last word of each sequence)
#     labels = batch['labels']
#     # Remove the labels from the input tensor
#     net_inputs = batch['embedding']
#
#     # Eventually clear previous recorded gradients
#     optimizer.zero_grad()
#     # Forward pass
#     net_outs, _ = net(net_inputs)
#
#     # Update network
#     loss = loss_fn(net_outs, labels)
#     acc = acc_fn(net_outs, labels)
#     # Backward pass
#     loss.backward()
#     # Update
#     optimizer.step()
#     return loss.data, acc
#
#
# def val_batch(net, batch):
#     # Get the labels (the last word of each sequence)
#     labels = batch['label']
#     # Remove the labels from the input tensor
#     net_inputs = batch['embedding']
#     # evaluate the network over the input
#     net_outs, _ = net(net_inputs)
#
#     return net_outs, labels
