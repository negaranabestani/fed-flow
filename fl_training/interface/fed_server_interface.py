from abc import ABC, abstractmethod

import torch
import logging
import torchvision
import torchvision.transforms as transforms
from config import config
from entity.Communicator import Communicator
from util import fl_utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FedServerInterface(ABC, Communicator):
    def __init__(self, index, ip_address, server_port, model_name):
        super(FedServerInterface, self).__init__(index, ip_address)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.port = server_port
        self.model_name = model_name
        self.sock.bind((self.ip, self.port))
        self.client_socks = {}

        while len(self.client_socks) < config.K:
            self.sock.listen(5)
            logger.info("Waiting Incoming Connections.")
            (client_sock, (ip, port)) = self.sock.accept()
            logger.info('Got connection from ' + str(ip))
            logger.info(client_sock)
            self.client_socks[str(ip)] = client_sock

        self.uninet = fl_utils.get_model('Unit', self.model_name, config.model_len - 1, self.device, config.model_cfg)

        self.transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
             ])
        self.testset = torchvision.datasets.CIFAR10(root=config.dataset_path, train=False, download=False,
                                                    transform=self.transform_test)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=100, shuffle=False, num_workers=4)

    @abstractmethod
    def initialize(self, split_layers, offload, first, LR):
        pass

    @abstractmethod
    def train(self, thread_number, client_ips):
        pass

    @abstractmethod
    def aggregate(self, client_ips):
        pass

    def scatter(self, msg):
        for i in self.client_socks:
            self.send_msg(self.client_socks[i], msg)
