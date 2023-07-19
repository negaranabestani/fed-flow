import logging
from abc import ABC, abstractmethod

import torch

from config import config
from entity.Communicator import Communicator
from util import fl_utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FedClientInterface(ABC, Communicator):
    def __init__(self, index, ip_address, server_addr, server_port, datalen, model_name, split_layer):
        super(FedClientInterface, self).__init__(index, ip_address)
        self.datalen = datalen
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.uninet = fl_utils.get_model('Unit', self.model_name, config.model_len - 1, self.device, config.model_cfg)

        logger.info('Connecting to Server.')
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
