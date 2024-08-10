import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

from app.dto.message import GlobalWeightMessage, JsonMessage, NetworkTestMessage, IterationFlagMessage, \
    SplitLayerConfigMessage
from app.entity.communicator import Communicator
from app.entity.fed_base_node_interface import FedBaseNodeInterface
from app.entity.node import Node
from app.entity.node_type import NodeType

sys.path.append('../../')
from app.util import model_utils, data_utils
from app.config.logger import fed_logger
from app.util.energy_estimation import *

np.random.seed(0)
torch.manual_seed(0)


# noinspection PyTypeChecker
class Client(FedBaseNodeInterface):
    def __init__(self, ip: str, port: int, model_name, dataset,
                 train_loader, LR, edge_based):
        Node.__init__(self, ip, port, NodeType.CLIENT)
        Communicator.__init__(self)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.edge_based = edge_based
        self.dataset = dataset
        self.train_loader = train_loader
        self.split_layers = None
        self.net = {}
        self.uninet = model_utils.get_model('Unit', None, self.device, edge_based)
        # self.uninet = model_utils.get_model('Unit', config.split_layer[config.index], self.device, edge_based)
        self.net = self.uninet
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=LR, momentum=0.9)

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
        edge_exchange = 'edge.' + config.CLIENTS_INDEX_TO_NAME[config.index]
        msg = GlobalWeightMessage([self.net.cpu().state_dict()])
        self.send_msg(edge_exchange, config.mq_url, msg)
        return msg

    def send_local_weights_to_server(self):
        msg = GlobalWeightMessage([self.net.cpu().state_dict()])
        self.send_msg(config.SERVER_INDEX_TO_NAME[config.index], config.mq_url, msg)
        return msg

    def test_network(self):
        """
        send message to test network speed
        """
        _ = self.recv_msg(config.CLIENTS_INDEX_TO_NAME[config.index],
                          config.mq_url,
                          NetworkTestMessage.MESSAGE_TYPE)
        fed_logger.info("test network received")
        msg = NetworkTestMessage([self.uninet.cpu().state_dict()])
        server_exchange = 'server.' + config.CLIENTS_INDEX_TO_NAME[config.index]
        self.send_msg(server_exchange, config.mq_url, msg)
        fed_logger.info("test network sent")
        return msg

    def edge_test_network(self):
        """
        send message to test network speed
        """
        _ = self.recv_msg(config.CLIENTS_INDEX_TO_NAME[config.index],
                          config.mq_url,
                          NetworkTestMessage.MESSAGE_TYPE)

        fed_logger.info("test network received")
        msg = NetworkTestMessage([self.uninet.cpu().state_dict()])
        self.send_msg(config.CLIENT_NAME_TO_EDGE_NAME[config.CLIENTS_INDEX_TO_NAME[config.index]], config.mq_url, msg)
        fed_logger.info("test network sent")
        return msg

    def get_split_layers_config(self):
        """
        receive splitting data
        """
        self.split_layers = self.recv_msg(config.CLIENTS_INDEX_TO_NAME[config.index], config.mq_url,
                                          SplitLayerConfigMessage.MESSAGE_TYPE).data

    def get_split_layers_config_from_edge(self):
        """
        receive splitting data
        """
        self.split_layers = self.recv_msg(config.CLIENTS_INDEX_TO_NAME[config.index], config.mq_url,
                                          SplitLayerConfigMessage.MESSAGE_TYPE).data

    def get_edge_global_weights(self):
        """
        receive global weights
        """
        msg: GlobalWeightMessage = \
            self.recv_msg(config.CLIENTS_INDEX_TO_NAME[config.index], config.mq_url, GlobalWeightMessage.MESSAGE_TYPE)
        pweights = model_utils.split_weights_client(msg.weights[0], self.net.state_dict())
        self.net.load_state_dict(pweights)

    def get_server_global_weights(self):
        """
        receive global weights
        """
        msg: GlobalWeightMessage = \
            self.recv_msg(config.CLIENTS_INDEX_TO_NAME[config.index], config.mq_url, GlobalWeightMessage.MESSAGE_TYPE)
        pweights = model_utils.split_weights_client(msg.weights[0], self.net.state_dict())
        self.net.load_state_dict(pweights)

    def edge_offloading_train(self):
        computation_start()
        self.net.to(self.device)
        self.net.train()
        computation_end()
        i = 0
        if self.split_layers[config.index][0] == model_utils.get_unit_model_len() - 1:
            fed_logger.info("no offloding training start----------------------------")
            flag = False
            start_transmission()
            self.send_msg(config.CLIENTS_INDEX_TO_NAME[config.index], config.mq_url, IterationFlagMessage(flag))
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
            edge_exchange = 'edge.' + config.CLIENTS_INDEX_TO_NAME[config.index]
            client_exchange = 'client.' + config.CLIENTS_INDEX_TO_NAME[config.index]
            # flag = [message_utils.local_iteration_flag_client_to_edge(), True]
            fed_logger.info(f"offloding training start {self.split_layers}----------------------------")
            flag = True
            start_transmission()
            self.send_msg(edge_exchange, config.mq_url, IterationFlagMessage(flag))
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
                flag = True
                start_transmission()
                self.send_msg(edge_exchange, config.mq_url, IterationFlagMessage(flag))
                end_transmission(data_utils.sizeofmessage(flag))

                msg = GlobalWeightMessage([outputs.cpu(),
                                           targets.cpu()])
                start_transmission()
                self.send_msg(edge_exchange, config.mq_url, msg)
                end_transmission(data_utils.sizeofmessage(msg))

                # Wait receiving edge server gradients
                fed_logger.info("receiving gradients")
                msg: GlobalWeightMessage = self.recv_msg(client_exchange,
                                                         config.mq_url,
                                                         GlobalWeightMessage.MESSAGE_TYPE)
                gradients = msg.weights[0].to(
                    self.device)
                # fed_logger.info("received gradients")
                computation_start()
                outputs.backward(gradients)
                if self.optimizer is not None:
                    self.optimizer.step()
                computation_end()
                i += 1
            flag = False
            start_transmission()
            self.send_msg(edge_exchange, config.mq_url, IterationFlagMessage(flag))
            end_transmission(data_utils.sizeofmessage(flag))

    def offloading_train(self):
        self.net.to(self.device)
        self.net.train()
        flag = True
        start_transmission()
        self.send_msg(config.CLIENTS_INDEX_TO_NAME[config.index], config.mq_url, IterationFlagMessage(flag))
        end_transmission(data_utils.sizeofmessage(flag))
        for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(self.train_loader)):
            flag = True
            start_transmission()
            self.send_msg(config.CLIENTS_INDEX_TO_NAME[config.index], config.mq_url, IterationFlagMessage(flag))
            end_transmission(data_utils.sizeofmessage(flag))
            computation_start()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            if self.optimizer is not None:
                self.optimizer.zero_grad()
            outputs = self.net(inputs)
            # fed_logger.info("sending local activations")
            msg = GlobalWeightMessage([outputs.cpu(), targets.cpu()])
            computation_end()
            start_transmission()
            self.send_msg(config.CLIENTS_INDEX_TO_NAME[config.index], config.mq_url, msg)
            end_transmission(data_utils.sizeofmessage(msg))

            # Wait receiving edge server gradients
            # fed_logger.info("receiving gradients")
            msg: GlobalWeightMessage = self.recv_msg(config.CLIENTS_INDEX_TO_NAME[config.index],
                                                     config.mq_url, GlobalWeightMessage.MESSAGE_TYPE)
            gradients = msg.weights[0].to(
                self.device)
            computation_start()
            outputs.backward(gradients)
            if self.optimizer is not None:
                self.optimizer.step()
            computation_end()

        flag = False
        start_transmission()
        self.send_msg(config.CLIENTS_INDEX_TO_NAME[config.index], config.mq_url, IterationFlagMessage(flag))
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
        msg = JsonMessage([energy, tt])
        self.send_msg(config.CLIENTS_INDEX_TO_NAME[config.index], config.mq_url, msg)
