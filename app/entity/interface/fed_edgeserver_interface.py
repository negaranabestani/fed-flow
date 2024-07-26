from abc import ABC, abstractmethod

import torch
from torch import multiprocessing

from app.config import config
from app.dto.message import BaseMessage
from app.entity.communicator import Communicator
from app.entity.node import Node
from app.util import data_utils, model_utils


class FedEdgeServerInterface(Node, ABC, Communicator):
    def __init__(self, ip: str, port: int, model_name, dataset, offload):
        Node.__init__(self, ip, port, 'Edge')
        Communicator.__init__(self)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.nets = {}
        self.group_labels = None
        self.criterion = None
        self.split_layers = None
        self.state = None
        self.client_bandwidth = {}
        self.dataset = dataset
        self.threads = None
        self.net_threads = None
        self.central_server_communicator = Communicator()

        if offload:
            model_len = model_utils.get_unit_model_len()
            self.uninet = model_utils.get_model('Unit', [model_len - 1, model_len - 1], self.device, True)

            self.testset = data_utils.get_testset()
            self.testloader = data_utils.get_testloader(self.testset, multiprocessing.cpu_count())

    @abstractmethod
    def test_client_network(self, client_ips):
        """
        send message to test_app network speed
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
    def get_split_layers_config(self, client_ips):
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
    def get_global_weights(self, client_ips: []):
        """
        receive global weights
        """
        pass

    @abstractmethod
    def energy(self, client_ip):
        """
        Returns: client energy consumption of the given client_ip
        """
        pass

    @abstractmethod
    def initialize(self, split_layers, LR, client_ips):
        pass

    def scatter(self, msg: BaseMessage):
        for i in config.EDGE_NAME_TO_CLIENTS_NAME[config.EDGE_SERVER_INDEX_TO_NAME[config.index]]:
            self.send_msg(i, config.mq_url, msg)

    @abstractmethod
    def forward_propagation(self, client_ip):
        pass

    @abstractmethod
    def backward_propagation(self, outputs, client_ip, inputs):
        pass

    @abstractmethod
    def thread_offload_training(self, client_ip):
        pass

    @abstractmethod
    def thread_no_offload_training(self, client_ip):
        pass

    @abstractmethod
    def no_offload_global_weights(self):
        pass
