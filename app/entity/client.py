import os
import socket
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

from app.config import config

sys.path.append('../../')
from app.util import message_utils, model_utils
from app.entity.interface.fed_client_interface import FedClientInterface
from app.config.logger import fed_logger
from app.util.energy_estimation import *

np.random.seed(0)
torch.manual_seed(0)


class Client(FedClientInterface):

    def initialize(self, split_layer, LR):

        self.split_layers = split_layer

        fed_logger.debug('Building Model.')
        self.net = model_utils.get_model('Client', self.split_layers[config.index], self.device, self.edge_based)
        fed_logger.debug(self.net)
        self.criterion = nn.CrossEntropyLoss()
        if len(list(self.net.parameters())) != 0:
            self.optimizer = optim.SGD(self.net.parameters(), lr=LR,
                                       momentum=0.9)

    def edge_upload(self):
        msg = [message_utils.local_weights_client_to_edge, self.net.cpu().state_dict()]
        self.send_msg(self.sock, msg)
        return msg

    def server_upload(self):
        msg = [message_utils.local_weights_client_to_server, self.net.cpu().state_dict()]
        self.send_msg(self.sock, msg)

    def test_network(self):
        """
        send message to test network speed
        """
        msg = self.recv_msg(self.sock, message_utils.test_network)[1]
        msg = [message_utils.test_network, self.uninet.cpu().state_dict()]
        self.send_msg(self.sock, msg)

    def split_layer(self):
        """
        receive splitting data
        """
        self.split_layers = self.recv_msg(self.sock, message_utils.split_layers)[1]

    def edge_global_weights(self):
        """
        receive global weights
        """
        weights = self.recv_msg(self.sock, message_utils.initial_global_weights_edge_to_client)[1]
        pweights = model_utils.split_weights_client(weights, self.net.state_dict())
        self.net.load_state_dict(pweights)

    def server_global_weights(self):
        """
        receive global weights
        """
        weights = self.recv_msg(self.sock, message_utils.initial_global_weights_server_to_client)[1]
        pweights = model_utils.split_weights_client(weights, self.net.state_dict())
        self.net.load_state_dict(pweights)

    def edge_offloading_train(self):

        flag = [message_utils.local_iteration_flag_client_to_edge, True]
        start_transmission()
        self.send_msg(self.sock, flag)
        end_transmission(sys.getsizeof(flag)*8)
        for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(self.train_loader)):
            flag = [message_utils.local_iteration_flag_client_to_edge, True]
            start_transmission()
            self.send_msg(self.sock, flag)
            end_transmission(sys.getsizeof(flag)*8)
            computation_start()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            if self.optimizer is not None:
                self.optimizer.zero_grad()
            outputs = self.net(inputs)
            computation_end()
            # fed_logger.info("sending local activations")
            msg = [message_utils.local_activations_client_to_edge, outputs.cpu(), targets.cpu()]
            start_transmission()
            self.send_msg(self.sock, msg)
            end_transmission(sys.getsizeof(msg)*8)

            # Wait receiving edge server gradients
            # fed_logger.info("receiving gradients")
            gradients = self.recv_msg(self.sock, message_utils.server_gradients_edge_to_client + socket.gethostname())[
                1].to(
                self.device)
            computation_start()
            outputs.backward(gradients)
            if self.optimizer is not None:
                self.optimizer.step()
            computation_end()

        flag = [message_utils.local_iteration_flag_client_to_edge, False]
        start_transmission()
        self.send_msg(self.sock, flag)
        end_transmission(sys.getsizeof(flag)*8)

    def offloading_train(self):
        flag = [message_utils.local_iteration_flag_client_to_server + '_' + socket.gethostname(), True]
        self.send_msg(self.sock, flag)
        for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(self.train_loader)):
            flag = [message_utils.local_iteration_flag_client_to_server + '_' + socket.gethostname(), True]
            self.send_msg(self.sock, flag)
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            if self.optimizer is not None:
                self.optimizer.zero_grad()
            outputs = self.net(inputs)
            # fed_logger.info("sending local activations")
            msg = [message_utils.local_activations_client_to_server + '_' + socket.gethostname(), outputs.cpu(),
                   targets.cpu()]
            self.send_msg(self.sock, msg)

            # Wait receiving edge server gradients
            # fed_logger.info("receiving gradients")
            gradients = \
                self.recv_msg(self.sock, message_utils.server_gradients_server_to_client + socket.gethostname())[
                    1].to(
                    self.device)

            outputs.backward(gradients)
            if self.optimizer is not None:
                self.optimizer.step()

        flag = [message_utils.local_iteration_flag_client_to_server + '_' + socket.gethostname(), False]
        self.send_msg(self.sock, flag)

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

    def energy_tt(self, energy,tt):
        msg = [message_utils.energy_client_to_edge + '_' + socket.gethostname(), energy,tt]
        self.send_msg(self.sock, msg)
