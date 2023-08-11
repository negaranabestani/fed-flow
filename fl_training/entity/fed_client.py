import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

sys.path.append('../../')
from config import config
from util import model_utils, message_utils
from fl_training.interface.fed_client_interface import FedClientInterface
from config.logger import fed_logger

np.random.seed(0)
torch.manual_seed(0)


class Client(FedClientInterface):

    def initialize(self, split_layer, LR):

        self.split_layers = split_layer

        fed_logger.debug('Building Model.')
        self.net = model_utils.get_model('Client', self.split_layer, self.device)
        fed_logger.debug(self.net)
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(self.net.parameters(), lr=LR,
                                   momentum=0.9)

    def edge_upload(self):
        msg = [message_utils.local_weights_client_to_server, self.net.cpu().state_dict()]
        self.send_msg(self.sock, msg)

    def server_upload(self):
        msg = [message_utils.local_weights_client_to_edge, self.net.cpu().state_dict()]
        self.send_msg(self.sock, msg)

    def test_network(self):
        """
        send message to test network speed
        """
        msg = self.recv_msg(self.sock, message_utils.test_client_network)[1]
        msg = [message_utils.test_client_network, self.uninet.cpu().state_dict()]
        self.send_msg(self.sock, msg)

    def split_layer(self):
        """
        receive splitting data
        """
        self.split_layers = self.recv_msg(self.sock, message_utils.split_layers_edge_to_client)[1]

    def edge_global_weights(self):
        """
        send final weights for aggregation
        """
        fed_logger.debug('Receiving Global Weights..')
        weights = self.recv_msg(self.sock, message_utils.initial_global_weights_edge_to_client)[1]
        pweights = model_utils.split_weights_client(weights, self.net.state_dict())
        self.net.load_state_dict(pweights)
        fed_logger.debug('Initialize Finished')

    def server_global_weights(self):
        """
        send final weights for aggregation
        """
        fed_logger.debug('Receiving Global Weights..')
        weights = self.recv_msg(self.sock, message_utils.initial_global_weights_server_to_client)[1]
        pweights = model_utils.split_weights_client(weights, self.net.state_dict())
        self.net.load_state_dict(pweights)
        fed_logger.debug('Initialize Finished')

    def offloading_train(self):
        for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(self.train_loader)):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)

            msg = [message_utils.local_activations_client_to_edge, outputs.cpu(), targets.cpu()]
            self.send_msg(self.sock, msg)

            # Wait receiving edge server gradients
            gradients = self.recv_msg(self.sock)[1].to(self.device)

            outputs.backward(gradients)
            self.optimizer.step()

    def no_offloading_train(self):
        self.net.to(self.device)
        self.net.train()
        for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(self.train_loader)):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
