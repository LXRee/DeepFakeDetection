# -*- coding: utf-8 -*-

from torch import nn
import torch


class Network(nn.Module):

    def __init__(self,
                 hidden_units,
                 layers_num,
                 embedding_dim=512,
                 dropout_prob=0.):
        # Call the parent init function (required!)
        super().__init__()

        # Define recurrent layer
        self.rnn = nn.LSTM(input_size=embedding_dim,
                           hidden_size=hidden_units,
                           num_layers=layers_num,
                           dropout=dropout_prob,
                           batch_first=True)
        # Define output layer
        self.out = nn.Linear(hidden_units, 2)

    def forward(self, inputs, state=None):
        # LSTM
        x, rnn_state = self.rnn(inputs, state)
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
