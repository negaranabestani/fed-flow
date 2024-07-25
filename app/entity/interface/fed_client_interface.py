from abc import ABC, abstractmethod

import torch
from torch import nn, optim

from app.entity.aggregator_config import AggregatorConfig
from app.entity.communicator import Communicator
from app.entity.node import Node
from app.util import model_utils


class FedClientInterface(Node, ABC, Communicator):
    def __init__(self, ip: str, port: int, server, datalen, model_name, dataset,
                 train_loader, LR, edge_based):
        Communicator.__init__(self)

        self.datalen = datalen
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.edge_based = edge_based
        self.server_id = server
        self.dataset = dataset
        self.train_loader = train_loader
        self.split_layers = None
        self.net = {}
        self.uninet = model_utils.get_model('Unit', None, self.device, edge_based)
        # self.uninet = model_utils.get_model('Unit', config.split_layer[config.index], self.device, edge_based)
        self.net = self.uninet
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=LR,
                                   momentum=0.9)

        Node.__init__(self, ip, port, aggregator_config=AggregatorConfig(
            self.uninet, self.split_layers, self.net, edge_based, False)
                      )

    @abstractmethod
    def initialize(self, split_layer, LR):
        pass

    @abstractmethod
    def send_local_weights_to_edge(self):
        """
        send final weights for aggregation
        """
        pass

    @abstractmethod
    def send_local_weights_to_server(self):
        pass

    @abstractmethod
    def test_network(self):
        """
        send message to test_app network speed
        """
        pass

    @abstractmethod
    def get_server_global_weights(self):
        pass

    def no_offloading_train(self):
        pass

    def edge_test_network(self):
        pass

    @abstractmethod
    def get_split_layers_config_from_edge(self):
        pass

    @abstractmethod
    def get_split_layers_config(self):
        """
        receive splitting data
        """
        pass

    @abstractmethod
    def get_edge_global_weights(self):
        """
        receive global weights
        """
        pass

    @abstractmethod
    def get_server_global_weights(self):
        pass

    @abstractmethod
    def edge_offloading_train(self):
        pass

    @abstractmethod
    def no_offloading_train(self):
        pass

    @abstractmethod
    def offloading_train(self):
        pass

    @abstractmethod
    def energy_tt(self, energy, tt):
        pass
