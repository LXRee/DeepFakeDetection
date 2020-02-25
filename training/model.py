import torch
from torch import nn
from torch.utils.data import DataLoader

from typing import Dict, Union
import os

from source.training.network import Network
from source.training.pytorchtools import EarlyStopping, Accuracy

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class Model:
    def __init__(self,
                 net_params: Dict[str, Union[str, int]],
                 pos_weights,
                 optimizer: str = "adam",
                 loss: str = "crossentropy",
                 lr: float = .1,
                 embedding_dim: int = 512):
        self.__net_params = net_params
        self.__optimizer_type = optimizer
        self.__loss_type = loss
        self.__pos_weights = pos_weights
        self.embedding_dim = embedding_dim
        self.learning_rate = lr

        self.acc, self.loss, self.optimizer, self.net = self.__build_model()

        self.train_loss_log = torch.tensor([]).to(device)
        self.val_loss_log = torch.tensor([]).to(device)
        self.test_loss_log = torch.tensor([]).to(device)

        # to restore optimizer's state when evaluating
        self.last_epoch = 0

    def __build_model(self) -> (torch.Tensor, torch.optim, nn.Module):
        """
        Metorchod to build torche network, torche loss and its optimizer.
        Return: loss, optimizer, network
        """
        hidden_units, layers_num, dropout_prob = self.__net_params['hidden_units'], self.__net_params['layers_num'], \
                                                 self.__net_params['dropout_prob']
        network: nn.Module = Network(
            hidden_units,
            layers_num,
            self.embedding_dim,
            dropout_prob
        )

        # send network to torche right device
        network.to(device)
        acc_fn = Accuracy()
        if self.__loss_type == "BCE":
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.__pos_weights).double())
        else:
            raise ValueError("{} loss is not implemented yet".format(self.__loss_type))

        if self.__optimizer_type == "SGD":
            optimizer = torch.optim.SGD(network.parameters(), lr=self.learning_rate, momentum=0.9)
        elif self.__optimizer_type == 'adam':
            optimizer = torch.optim.Adam(network.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        else:
            raise ValueError("{} is not implemented yet".format(self.__optimizer_type))

        return acc_fn, loss_fn, optimizer, network

    def save(self, run_path: str = "exp0", loss_value: float = 10):
        """
        Saves only torche model parameters. To restore torche training it is better to see "save"
        """
        self.net: nn.Module
        state = {
            'epoch': self.last_epoch + 1,
            'state_dict': self.net.state_dict(),
            # 'optim_dict': self.optimizer.state_dict()
        }
        self.net: nn.Module
        torch.save(state, os.path.join(run_path, 'checkpoint_' + str(loss_value) + '.pt'))

    def fit(self, epochs: int, train_set: DataLoader, val_set: DataLoader, patience: int = 20, run_name: str = None):
        self.net: torch.nn.Module
        early_stopping = EarlyStopping(patience=patience, verbose=False)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, len(train_set), 1e-7)
        net = self.net
        optimizer = self.optimizer
        loss_fn = self.loss
        acc_fn = self.acc

        for epoch in range(epochs):
            net.train()
            conc_losses: torch.Tensor = torch.tensor([]).to(device)
            conc_acc: torch.Tensor = torch.tensor([]).to(device)

            for i, batch in enumerate(train_set):
                net_inputs = batch['embedding'].to(device)
                # batch input comes as sparse
                # Get the labels (the last word of each sequence)
                labels = batch['label'].to(device)
                # Forward pass
                net_outs, _ = net(net_inputs)
                # Update network
                loss = loss_fn(net_outs, labels)
                acc = acc_fn(net_outs, labels)

                # Eventually clear previous recorded gradients
                optimizer.zero_grad()
                # Backward pass
                loss.backward()
                # clip gradients to avoid gradient explosion
                # torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                # Update
                optimizer.step()
                # return loss.data, acc
                # scheduler.step(epoch)
                conc_losses = torch.cat([conc_losses, torch.unsqueeze(loss, dim=-1)])
                conc_acc = torch.cat([conc_acc, torch.unsqueeze(acc, dim=-1)])
            # update scheduler outside batch loop
            scheduler.step(epoch)
            epoch_train_loss = torch.mean(conc_losses)
            epoch_train_acc = torch.mean(conc_acc)

            # validation
            net.eval()
            conc_out: torch.Tensor = torch.tensor([]).to(device)
            conc_label: torch.Tensor = torch.tensor([]).to(device)
            with torch.no_grad():
                for i, batch in enumerate(val_set):
                    net_inputs = batch['embedding'].to(device)
                    labels = batch['label'].to(device)

                    # evaluate the network over the input
                    net_outs, _ = net(net_inputs)

                    conc_out = torch.cat([conc_out, torch.unsqueeze(net_outs, dim=-1)])
                    conc_label = torch.cat([conc_label, torch.unsqueeze(labels, dim=-1)])
                epoch_val_loss = self.loss(conc_out, conc_label)
                epoch_val_acc = self.acc(conc_out, conc_label).float()

            # if epoch % 1 == 0:
            print("Epoch: {}\ttrain: acc: {:4f} loss: {:.4f}\t\tval: acc: {:.4f} loss: {:.4f}".format(epoch,
                                                                                                      epoch_train_acc,
                                                                                                      epoch_train_loss,
                                                                                                      epoch_val_acc,
                                                                                                      epoch_val_loss))

            if epoch % 10 == 0:
                early_stopping(epoch_val_loss, self.net)
                if early_stopping.save_checkpoint and run_name:
                    self.save(run_name, epoch_val_loss.cpu().detach().numpy())
                if early_stopping.early_stop:
                    print("Early stopping")
                    self.last_epoch = epoch
                    break
