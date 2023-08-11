from abc import ABC, abstractmethod

import torch

from config import config
from config.logger import fed_logger
from entity.Communicator import Communicator
from util import model_utils


class FedClientInterface(ABC, Communicator):
    def __init__(self, index, ip_address, server_addr, server_port, datalen, model_name, dataset, split_layers,
                 train_loader):
        super(FedClientInterface, self).__init__(index, ip_address)
        self.datalen = datalen
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.dataset = dataset
        self.train_loader = train_loader
        self.split_layers = split_layers
        self.uninet = model_utils.get_model('Unit', config.model_len - 1, self.device)

        fed_logger.info('Connecting to Server.')
        self.sock.connect((server_addr, server_port))

    @abstractmethod
    def initialize(self, split_layer, offload, first, LR):
        pass

    @abstractmethod
    def upload(self):
        pass

    @abstractmethod
    def test_network(self):
        """
        send message to test network speed
        """
        pass

    @abstractmethod
    def split_layer(self):
        """
        receive splitting data
        """
        pass

    @abstractmethod
    def local_weights(self, client_ip):
        """
        send final weights for aggregation
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
    def forward_propagation(self):
        pass

    @abstractmethod
    def backward_propagation(self, outputs):
        pass
