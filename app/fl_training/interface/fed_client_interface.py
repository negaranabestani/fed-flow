from abc import ABC, abstractmethod

import torch
from torch import nn, optim

from app.config.logger import fed_logger
from app.entity.Communicator import Communicator
from app.util import model_utils


class FedClientInterface(ABC, Communicator):
    def __init__(self, index, ip_address, server_addr, server_port, datalen, model_name, dataset,
                 train_loader, LR):
        super(FedClientInterface, self).__init__(index, ip_address)
        self.datalen = datalen
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.dataset = dataset
        self.train_loader = train_loader
        self.split_layers = None
        self.net = {}
        model_len = model_utils.get_unit_model_len()
        self.uninet = model_utils.get_model('Unit', [model_len - 1, model_len - 1], self.device)
        self.net = self.uninet
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=LR,
                                   momentum=0.9)

        fed_logger.info('Connecting to Server.')
        self.sock.connect((server_addr, server_port))

    @abstractmethod
    def initialize(self, split_layer, LR):
        pass

    @abstractmethod
    def edge_upload(self):
        """
        send final weights for aggregation
        """
        pass

    @abstractmethod
    def server_upload(self):
        pass

    @abstractmethod
    def test_network(self):
        """
        send message to test_app network speed
        """
        pass

    @abstractmethod
    def split_layer(self):
        """
        receive splitting data
        """
        pass

    @abstractmethod
    def edge_global_weights(self):
        """
        receive global weights
        """
        pass

    @abstractmethod
    def server_global_weights(self):
        pass

    @abstractmethod
    def offloading_train(self):
        pass

    @abstractmethod
    def no_offloading_train(self):
        pass
