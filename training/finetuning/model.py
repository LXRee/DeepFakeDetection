import os
from typing import Dict, Union

import torch
from torch import nn
from torch.utils.data import DataLoader

from training.finetuning.network import LSTMNetwork, TransformerNetwork
from training.pytorchtools import EarlyStopping, DeepFakeMetric

use_cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if use_cuda else "cpu")


class Model:
    def __init__(self,
                 network_type: str,
                 net_params: Dict[str, Union[str, int]],
                 optimizer: str = "adam",
                 loss: str = "crossentropy",
                 lr: float = .1):

        self.__network_type = network_type
        self.__net_params = net_params
        self.__optimizer_type = optimizer
        self.__loss_type = loss
        self.learning_rate = lr

        self.acc, self.loss, self.optimizer, self.net = self.__build_model()

    def __build_model(self) -> (torch.Tensor, torch.optim, nn.Module):
        """
        Build Torch network, loss and its optimizer.
        @:return loss, optimizer, network
        """
        dropout_prob = self.__net_params['dropout_prob']
        video_emb_dim = self.__net_params['video_embedding_dim']
        audio_emb_dim = self.__net_params['audio_embedding_dim']
        fc_dim = self.__net_params['fc_dim']

        # choose hyperparameters for the right network and define network
        if self.__network_type == 'LSTM':
            hidden_units = self.__net_params['hidden_units']
            layers_num = self.__net_params['layers_num']
            network: nn.Module = LSTMNetwork(hidden_units,
                                             layers_num,
                                             fc_dim,
                                             video_emb_dim,
                                             audio_emb_dim,
                                             dropout_prob)
        elif self.__network_type == 'transformer':
            n_head = self.__net_params['n_head']
            dim_feedforward = self.__net_params['dim_feedforward']
            enc_layers = self.__net_params['enc_layers']
            network: nn.Module = TransformerNetwork(n_head,
                                                    dim_feedforward,
                                                    enc_layers,
                                                    dropout_prob,
                                                    fc_dim,
                                                    video_emb_dim,
                                                    audio_emb_dim,
                                                    )
        else:
            raise ValueError('Bad network type. Please choose between "LSTM" and "transfomer"')

        # Add pretrained weights to network
        weights = torch.load(self.__net_params['checkpoint_path'])['state_dict']
        delete_weights = [
            'concat_branches.weight',
            'concat_branches.bias',
            'out.weight',
            'out.bias'
        ]
        for w in delete_weights:
            weights.pop(w)
        network.load_state_dict(weights, strict=False)
        # Disable grads for all layers except last one
        for p in list(network.parameters())[:-2]:
            p.requires_grad = False

        network.to(DEVICE)
        acc_fn = DeepFakeMetric()

        if self.__loss_type == "crossentropy":
            loss_fn = nn.CrossEntropyLoss()
        elif self.__loss_type == 'BCE':
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            raise ValueError("{} loss is not implemented yet".format(self.__loss_type))

        if self.__optimizer_type == "SGD":
            optimizer = torch.optim.SGD(network.parameters(), lr=self.learning_rate, momentum=0.9)
        elif self.__optimizer_type == 'adam':
            optimizer = torch.optim.Adam(network.parameters(), lr=self.learning_rate, weight_decay=1e-6)
        else:
            raise ValueError("{} is not implemented yet".format(self.__optimizer_type))

        return acc_fn, loss_fn, optimizer, network

    def save(self, run_path: str = "exp0", metric: float = 10, epoch: int = 0):
        """
        Saves only Torch model parameters. To restore Torch training it is better to see "save"
        Save the state_dict only if you want to continue training from a certain point.
        It is really heavy since it contains the adaptive lr of Adam and is not useful for inference.
        """
        self.net: nn.Module
        state = {
            'state_dict': self.net.state_dict(),
            # 'optim_dict': self.optimizer.state_dict()
        }
        self.net: nn.Module
        filepath = os.path.join(run_path, 'checkpoint_' + str(metric) + '_ep' + str(epoch) + '.pt')
        torch.save(state, filepath)

    def fit(self, epochs: int, train_set: DataLoader, val_set: DataLoader, patience: int = 20, run_name: str = None):
        self.net: torch.nn.Module  # define type of self.net to ease linting
        early_stopping = EarlyStopping(patience=patience, verbose=False)
        # This scheduler let the learning rate to decrease each epoch, going from initial lr to the final (1e-7)
        # In this way, the loss function and the optimizer can speed up training by escaping local minima without
        # slowing down too much. This let us also keep higher lr values, since it is decreased in different way for
        # each image - since random provisioning is enabled.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, len(train_set), 1e-7)

        # Just for portability
        net = self.net
        optimizer = self.optimizer
        loss_fn = self.loss
        acc_fn = self.acc

        start_whole = torch.cuda.Event(enable_timing=True)
        end_whole = torch.cuda.Event(enable_timing=True)
        start_epoch = torch.cuda.Event(enable_timing=True)
        end_epoch = torch.cuda.Event(enable_timing=True)

        start_whole.record()
        for epoch in range(epochs):
            start_epoch.record()
            net.train()
            conc_losses = []
            conc_acc = []

            for batch in train_set:
                filename, video_embedding, audio_embedding, label = batch
                net_inputs = (video_embedding.to(DEVICE), audio_embedding.to(DEVICE))
                labels = label.to(DEVICE)

                # Forward pass
                net_outs = net(net_inputs).squeeze()

                # Update network
                loss = loss_fn(net_outs, labels)
                acc = acc_fn(net_outs, labels)

                # Eventually clear previous recorded gradients
                optimizer.zero_grad()

                # Backward pass
                loss.backward()

                # Clip gradients to avoid gradient explosion
                # torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)

                # Update
                optimizer.step()

                # Return loss, acc
                conc_losses.append(loss)
                conc_acc.append(acc)

            # Update scheduler outside batch loop
            scheduler.step(epoch)
            epoch_train_loss = torch.mean(torch.stack(conc_losses)).float()
            epoch_train_acc = torch.mean(torch.stack(conc_acc)).float()

            # Validation
            net.eval()
            conc_out = []
            conc_label = []
            with torch.no_grad():
                for batch in val_set:
                    filename, video_embedding, audio_embedding, label = batch
                    net_inputs = (video_embedding.to(DEVICE), audio_embedding.to(DEVICE))
                    labels = label.to(DEVICE)

                    # Evaluate the network over the input
                    net_outs = net(net_inputs).squeeze()

                    conc_out.append(net_outs)
                    conc_label.append(labels)

                conc_out = torch.cat(conc_out)
                conc_label = torch.cat(conc_label)
                epoch_val_loss = loss_fn(conc_out, conc_label).float()
                epoch_val_acc = acc_fn(conc_out, conc_label).float()

            end_epoch.record()
            torch.cuda.synchronize(DEVICE)
            print(
                "Epoch: {}\ttrain: acc: {:.4f} loss: {:.4f}\t\tval: acc: {:.4f} loss: {:.4f}\ttime: {:.4}s".format(
                    epoch,
                    epoch_train_acc,
                    epoch_train_loss,
                    epoch_val_acc,
                    epoch_val_loss,
                    start_epoch.elapsed_time(
                        end_epoch) / 1000))
            # Update early stopping. This is really useful to stop training in time.
            # The if statement is not slowing down training since each epoch last very long.
            # PLEASE TAKE NOTE THAT we are using epoch_val_acc, since it brings the score function of the competition
            float_epoch_val_acc = epoch_val_loss.detach().cpu().numpy()
            float_epoch_train_acc = epoch_train_loss.detach().cpu().numpy()
            early_stopping(float_epoch_train_acc, float_epoch_val_acc, self.net)
            if early_stopping.save_checkpoint and run_name:
                self.save(run_name, float_epoch_val_acc, epoch)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        end_whole.record()
        torch.cuda.synchronize(DEVICE)
        print("Elapsed time: {:.4f}s".format(start_whole.elapsed_time(end_whole) / 1000))

        # Return val_loss_min for KFold - which is, the metric that we register for early stopping.
        return early_stopping.val_loss_min
