import socket
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

sys.path.append('../../')
from app.util import message_utils, model_utils, data_utils
from app.entity.interface.fed_client_interface import FedClientInterface
from app.config.logger import fed_logger
from app.config import config
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

    def send_local_weights_to_edge(self):
        msg = [message_utils.local_weights_client_to_edge(), self.net.cpu().state_dict()]
        self.send_msg(config.CLIENTS_INDEX[config.index], msg, True)
        return msg

    def send_local_weights_to_server(self):
        msg = [message_utils.local_weights_client_to_server(), self.net.cpu().state_dict()]
        self.send_msg(config.CLIENTS_INDEX[config.index], msg, True)
        return msg

    def test_network(self):
        """
        send message to test network speed
        """
        msg = self.recv_msg(exchange=config.CLIENTS_INDEX[config.index],
                            expect_msg_type=message_utils.test_network_edge_to_client(), is_weight=True)[1]
        fed_logger.info("test network received")
        msg = [message_utils.test_network_client_to_edge(), self.uninet.cpu().state_dict()]
        self.send_msg(exchange=config.CLIENTS_INDEX[config.index], msg=msg, is_weight=True)
        fed_logger.info("test network sent")
        return msg

    def get_split_layers_config(self):
        """
        receive splitting data
        """
        self.split_layers = self.recv_msg(config.CLIENTS_INDEX[config.index], message_utils.split_layers())[1]

    def get_split_layers_config_from_edge(self):
        """
        receive splitting data
        """
        self.split_layers = \
            self.recv_msg(config.CLIENTS_INDEX[config.index], message_utils.split_layers_edge_to_client())[1]

    def get_edge_global_weights(self):
        """
        receive global weights
        """
        weights = \
            self.recv_msg(config.CLIENTS_INDEX[config.index], message_utils.initial_global_weights_edge_to_client(),
                          True)[1]
        pweights = model_utils.split_weights_client(weights, self.net.state_dict())
        self.net.load_state_dict(pweights)

    def get_server_global_weights(self):
        """
        receive global weights
        """
        weights = \
            self.recv_msg(config.CLIENTS_INDEX[config.index], message_utils.initial_global_weights_server_to_client(),
                          True)[1]
        pweights = model_utils.split_weights_client(weights, self.net.state_dict())
        self.net.load_state_dict(pweights)

    def edge_offloading_train(self):
        computation_start()
        self.net.to(self.device)
        self.net.train()
        computation_end()
        i = 0
        if self.split_layers[config.index][0] == model_utils.get_unit_model_len() - 1:
            fed_logger.info("no offloding training start----------------------------")
            flag = [f'{message_utils.local_iteration_flag_client_to_edge()}_{i}_{socket.gethostname()}', False]
            start_transmission()
            self.send_msg(config.CLIENTS_INDEX[config.index], flag)
            end_transmission(data_utils.sizeofmessage(flag))
            i += 1
            for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(self.train_loader)):
                computation_start()
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                computation_end()

        if self.split_layers[config.index][0] < model_utils.get_unit_model_len() - 1:
            # flag = [message_utils.local_iteration_flag_client_to_edge(), True]
            fed_logger.info(f"offloding training start {self.split_layers}----------------------------")
            flag = [f'{message_utils.local_iteration_flag_client_to_edge()}_{i}_{socket.gethostname()}', True]
            start_transmission()
            self.send_msg(config.CLIENTS_INDEX[config.index], flag)
            end_transmission(data_utils.sizeofmessage(flag))
            i += 1

            for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(self.train_loader)):

                computation_start()
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if self.optimizer is not None:
                    self.optimizer.zero_grad()
                outputs = self.net(inputs)
                computation_end()
                # fed_logger.info("sending local activations")
                flag = [f'{message_utils.local_iteration_flag_client_to_edge()}_{i}_{socket.gethostname()}', True]
                start_transmission()
                self.send_msg(config.CLIENTS_INDEX[config.index], flag)
                end_transmission(data_utils.sizeofmessage(flag))

                msg = [f'{message_utils.local_activations_client_to_edge()}_{i}_{socket.gethostname()}', outputs.cpu(),
                       targets.cpu()]
                # fed_logger.info(f"{msg[1], msg[2]}")
                start_transmission()
                self.send_msg(exchange=config.CLIENTS_INDEX[config.index], msg=msg, is_weight=True)
                end_transmission(data_utils.sizeofmessage(msg))

                # Wait receiving edge server gradients
                # fed_logger.info("receiving gradients")
                gradients = \
                    self.recv_msg(exchange=config.CLIENTS_INDEX[config.index],
                                  expect_msg_type=f'{message_utils.server_gradients_edge_to_client() + socket.gethostname()}_{i}',
                                  is_weight=True)[
                        1].to(
                        self.device)
                # fed_logger.info("received gradients")
                computation_start()
                outputs.backward(gradients)
                if self.optimizer is not None:
                    self.optimizer.step()
                computation_end()
                i += 1
            flag = [f'{message_utils.local_iteration_flag_client_to_edge()}_{i}_{socket.gethostname()}', False]
            start_transmission()
            self.send_msg(config.CLIENTS_INDEX[config.index], flag)
            end_transmission(data_utils.sizeofmessage(flag))

    def offloading_train(self):
        self.net.to(self.device)
        self.net.train()
        flag = [message_utils.local_iteration_flag_client_to_server() + '_' + socket.gethostname(), True]
        start_transmission()
        self.send_msg(config.CLIENTS_INDEX[config.index], flag)
        end_transmission(data_utils.sizeofmessage(flag))
        for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(self.train_loader)):
            flag = [message_utils.local_iteration_flag_client_to_server() + '_' + socket.gethostname(), True]
            start_transmission()
            self.send_msg(config.CLIENTS_INDEX[config.index], flag)
            end_transmission(data_utils.sizeofmessage(flag))
            computation_start()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            if self.optimizer is not None:
                self.optimizer.zero_grad()
            outputs = self.net(inputs)
            # fed_logger.info("sending local activations")
            msg = [message_utils.local_activations_client_to_server() + '_' + socket.gethostname(), outputs.cpu(),
                   targets.cpu()]
            computation_end()
            start_transmission()
            self.send_msg(config.CLIENTS_INDEX[config.index], msg, True)
            end_transmission(data_utils.sizeofmessage(msg))

            # Wait receiving edge server gradients
            # fed_logger.info("receiving gradients")
            gradients = \
                self.recv_msg(config.CLIENTS_INDEX[config.index],
                              message_utils.server_gradients_server_to_client() + socket.gethostname(), True)[
                    1].to(
                    self.device)
            computation_start()
            outputs.backward(gradients)
            if self.optimizer is not None:
                self.optimizer.step()
            computation_end()

        flag = [message_utils.local_iteration_flag_client_to_server() + '_' + socket.gethostname(), False]
        start_transmission()
        self.send_msg(config.CLIENTS_INDEX[config.index], flag)
        end_transmission(data_utils.sizeofmessage(flag))

    def no_offloading_train(self):
        self.net.to(self.device)
        self.net.train()
        for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(self.train_loader)):
            computation_start()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            computation_end()

    def energy_tt(self, energy, tt):
        msg = [message_utils.energy_client_to_edge() + '_' + socket.gethostname(), energy, tt]
        self.send_msg(config.CLIENTS_INDEX[config.index], msg)
