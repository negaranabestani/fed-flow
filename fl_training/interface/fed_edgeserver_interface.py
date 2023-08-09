from abc import ABC, abstractmethod

import torch
from torch import multiprocessing

from config import config
from config.logger import fed_logger
from entity.Communicator import Communicator
from util import model_utils, data_utils


class FedEdgeServerInterface(ABC, Communicator):
    def __init__(self, index, ip_address, server_port, model_name, dataset):
        super(FedEdgeServerInterface, self).__init__(index, ip_address)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.port = server_port
        self.model_name = model_name
        self.sock.bind((self.ip, self.port))
        self.socks = {}
        self.nets = {}
        self.group_labels = None
        self.criterion = None
        self.split_layers = None
        self.state = None
        self.client_bandwidth = {}
        self.dataset = dataset
        self.threads = None
        self.net_threads = None

        while len(self.socks) < len(config.EDGE_MAP[ip_address]):
            self.sock.listen(5)
            fed_logger.info("Waiting Incoming Connections.")
            (client_sock, (ip, port)) = self.sock.accept()
            fed_logger.info('Got connection from ' + str(ip))
            fed_logger.info(client_sock)
            self.socks[str(ip)] = client_sock

        self.uninet = model_utils.get_model('Unit', config.model_len - 1, self.device)

        self.testset = data_utils.get_testset()
        self.testloader = data_utils.get_testloader(self.testset, multiprocessing.cpu_count())

    @abstractmethod
    def test_client_network(self, client_ips):
        """
        send message to test network speed
        """
        pass

    @abstractmethod
    def test_server_network(self):
        pass

    @abstractmethod
    def client_network(self):
        """
        send client network speed to central server
        """
        pass

    @abstractmethod
    def split_layer(self):
        """
        receive send splitting data to clients
        """
        pass

    @abstractmethod
    def local_weights(self, client_ip):
        """
        receive and send final weights for aggregation
        """
        pass

    @abstractmethod
    def global_weights(self, client_ips: []):
        """
        receive global weights
        """
        pass

    @abstractmethod
    def initialize(self, split_layers, LR, client_ips):
        pass

    def scatter(self, msg):
        for i in self.socks:
            self.send_msg(self.socks[i], msg)

    @abstractmethod
    def forward_propagation(self, client_ip):
        pass

    @abstractmethod
    def backward_propagation(self, outputs, client_ip):
        pass

    @abstractmethod
    def thread_training(self, client_ip):
        pass
