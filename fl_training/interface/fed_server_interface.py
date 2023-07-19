from abc import ABC, abstractmethod

import torch
import torchvision
import torchvision.transforms as transforms

from config import config
from config.logger import fed_logger
from entity.Communicator import Communicator
from fl_method import fl_method_parser
from util import fl_utils


class FedServerInterface(ABC, Communicator):
    def __init__(self, index, ip_address, server_port, model_name, dataset):
        super(FedServerInterface, self).__init__(index, ip_address)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.port = server_port
        self.model_name = model_name
        self.sock.bind((self.ip, self.port))
        self.client_socks = {}
        self.group_labels = None
        self.criterion = None
        self.split_layers = None
        self.state = None
        self.bandwidth = None
        self.dataset = dataset

        while len(self.client_socks) < config.K:
            self.sock.listen(5)
            fed_logger.info("Waiting Incoming Connections.")
            (client_sock, (ip, port)) = self.sock.accept()
            fed_logger.info('Got connection from ' + str(ip))
            fed_logger.info(client_sock)
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
    def aggregate(self, client_ips, aggregate_method):
        pass

    def apply_options(self, options: dict):
        fl_method_parser.fl_methods.get(options.get('clustering'))()
        self.aggregate(config.CLIENTS_LIST, fl_method_parser.fl_methods.get(options.get('aggregation')))
        self.split_layers = fl_method_parser.fl_methods.get(options.get('splitting'))(self.state, self.group_labels)
        fed_logger.info('Next Round OPs: ' + str(config.split_layer))
        msg = ['SPLIT_LAYERS', config.split_layer]
        self.scatter(msg)

    def scatter(self, msg):
        for i in self.client_socks:
            self.send_msg(self.client_socks[i], msg)
