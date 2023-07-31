from abc import ABC, abstractmethod

import torch

from config import config
from config.logger import fed_logger
from entity.Communicator import Communicator
from util import model_utils


class FedClientInterface(ABC, Communicator):
    def __init__(self, index, ip_address, server_addr, server_port, datalen, model_name, dataset, split_layer):
        super(FedClientInterface, self).__init__(index, ip_address)
        self.datalen = datalen
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.dataset = dataset
        self.split_layer = split_layer
        self.uninet = model_utils.get_model('Unit', config.model_len - 1, self.device)

        fed_logger.info('Connecting to Server.')
        self.sock.connect((server_addr, server_port))

    @abstractmethod
    def initialize(self, split_layer, offload, first, LR):
        pass

    @abstractmethod
    def train(self, trainloader):
        pass

    @abstractmethod
    def upload(self):
        pass
