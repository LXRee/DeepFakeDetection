import os
from typing import Dict, Union

import torch
from torch import nn
from torch.utils.data import DataLoader

from training.network import Network
from training.pytorchtools import EarlyStopping, Accuracy

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class Model:
    def __init__(self,
                 net_params: Dict[str, Union[str, int]],
                 pos_weights,
                 optimizer: str = "adam",
                 loss: str = "crossentropy",
                 lr: float = .1):

        self.__net_params = net_params
        self.__optimizer_type = optimizer
        self.__loss_type = loss
        self.__pos_weights = pos_weights
        self.learning_rate = lr

        self.acc, self.loss, self.optimizer, self.net = self.__build_model()

        self.train_loss_log = torch.tensor([]).to(device)
        self.val_loss_log = torch.tensor([]).to(device)
        self.test_loss_log = torch.tensor([]).to(device)

        # TODO: To restore optimizer's state when training from previous step we need last epoch. I don't know where to save it
        self.last_epoch = 0

    def __build_model(self) -> (torch.Tensor, torch.optim, nn.Module):
        """
        Build Torch network, loss and its optimizer.
        @:return loss, optimizer, network
        """
        hidden_units = self.__net_params['hidden_units']
        layers_num = self.__net_params['layers_num']
        dropout_prob = self.__net_params['dropout_prob']
        video_emb_dim = self.__net_params['video_embedding_dim']
        audio_emb_dim = self.__net_params['audio_embedding_dim']

        network: nn.Module = Network(hidden_units,
                                     layers_num,
                                     video_emb_dim,
                                     audio_emb_dim,
                                     dropout_prob)

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
        Saves only Torch model parameters. To restore Torch training it is better to see "save"
        Save the state_dict only if you want to continue training from a certain point.
        It is really heavy since it contains the adaptive lr of Adam and is not useful for inference.
        """
        self.net: nn.Module
        state = {
            'epoch': self.last_epoch + 1,
            'state_dict': self.net.state_dict(),
            # 'optim_dict': self.optimizer.state_dict()
        }
        self.net: nn.Module
        filepath = os.path.join(run_path, 'checkpoint_' + str(loss_value) + '.pt')
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
            conc_losses: torch.Tensor = torch.tensor([]).to(device)
            conc_acc: torch.Tensor = torch.tensor([]).to(device)

            for batch in train_set:
                net_inputs = (batch['video_embedding'].to(device), batch['audio_embedding'].to(device))
                labels = batch['label'].to(device)

                # Forward pass
                net_outs, _ = net(net_inputs)

                # Update network
                loss = loss_fn(net_outs.squeeze(), labels)
                acc = acc_fn(net_outs.squeeze(), labels)

                # Eventually clear previous recorded gradients
                optimizer.zero_grad()

                # Backward pass
                loss.backward()

                # Clip gradients to avoid gradient explosion
                # torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)

                # Update
                optimizer.step()

                # Return loss, acc
                conc_losses = torch.cat([conc_losses, torch.unsqueeze(loss, dim=-1)])
                conc_acc = torch.cat([conc_acc, torch.unsqueeze(acc, dim=-1)])

            # Update scheduler outside batch loop
            scheduler.step(epoch)
            epoch_train_loss = torch.mean(conc_losses)
            epoch_train_acc = torch.mean(conc_acc)

            # Validation
            net.eval()
            conc_out: torch.Tensor = torch.tensor([]).to(device)
            conc_label: torch.Tensor = torch.tensor([]).to(device)
            with torch.no_grad():
                for batch in val_set:
                    net_inputs = (batch['video_embedding'].to(device), batch['audio_embedding'].to(device))
                    labels = batch['label'].to(device)

                    # Evaluate the network over the input
                    net_outs, _ = net(net_inputs)

                    conc_out = torch.cat([conc_out, torch.unsqueeze(net_outs.squeeze(), dim=-1)])
                    conc_label = torch.cat([conc_label, torch.unsqueeze(labels, dim=-1)])

                epoch_val_loss = loss_fn(conc_out, conc_label)
                epoch_val_acc = acc_fn(conc_out, conc_label).float()

            end_epoch.record()
            torch.cuda.synchronize(device)
            print(
                "Epoch: {}\ttrain: acc: {:4f} loss: {:.4f}\t\tval: acc: {:.4f} loss: {:.4f}\ttime: {:.4}s".format(epoch,
                                                                                                                  epoch_train_acc,
                                                                                                                  epoch_train_loss,
                                                                                                                  epoch_val_acc,
                                                                                                                  epoch_val_loss,
                                                                                                                  start_epoch.elapsed_time(end_epoch)/1000))
            # Update early stopping. This is really useful to stop training in time.
            # The if statement is not slowing down training since each epoch last very long.
            early_stopping(epoch_val_loss, self.net)
            if early_stopping.save_checkpoint and run_name:
                self.save(run_name, epoch_val_loss.cpu().detach().numpy())
            if early_stopping.early_stop:
                print("Early stopping")
                self.last_epoch = epoch
                break
        end_whole.record()
        torch.cuda.synchronize(device)
        print("Elapsed time: {:.4f}s".format(start_whole.elapsed_time(end_whole)/1000))

    def evaluate(self, test_set: DataLoader):
        self.net: torch.nn.Module
        net = self.net
        loss_fn = self.loss
        acc_fn = self.acc
        # Put network in evaluation mode aka Dropout and BatchNorm are disabled
        net.eval()
        with torch.no_grad():
            conc_out: torch.Tensor = torch.tensor([]).to(device)
            conc_label: torch.Tensor = torch.tensor([]).to(device)
            # Do not update gradients
            with torch.no_grad():
                for batch in test_set:
                    net_inputs = (batch['video_embedding'].to(device), batch['audio_embedding'].to(device))
                    labels = batch['label'].to(device)

                    # Evaluate the network over the input
                    net_outs, _ = net(net_inputs)

                    conc_out = torch.cat([conc_out, torch.unsqueeze(net_outs.squeeze(), dim=-1)])
                    conc_label = torch.cat([conc_label, torch.unsqueeze(labels, dim=-1)])

                epoch_test_loss = loss_fn(conc_out.squeeze(), conc_label.squeeze())
                epoch_test_acc = acc_fn(conc_out.squeeze(), conc_label.squeeze())

                print("Test:\tacc: {:.4f}\n\t\tloss: {:.4f}".format(epoch_test_acc, epoch_test_loss))
