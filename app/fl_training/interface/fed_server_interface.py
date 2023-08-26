import multiprocessing
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn

from app.config import config
from app.config.logger import fed_logger
from app.entity.Communicator import Communicator
from app.fl_method import fl_method_parser
from app.util import data_utils, model_utils


class FedServerInterface(ABC, Communicator):
    def __init__(self, index, ip_address, port, model_name, dataset,
                 connection_list):
        super(FedServerInterface, self).__init__(index, ip_address)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.port = port
        self.model_name = model_name
        self.sock.bind((self.ip, self.port))
        self.socks = {}
        self.group_labels = None
        self.criterion = None
        self.split_layers = None
        self.state = None
        self.client_bandwidth = {}
        self.edge_bandwidth = {}
        self.dataset = dataset
        self.threads = None
        self.net_threads = None
        self.offloading = None
        self.tt_start = {}
        self.tt_end = {}

        while len(self.socks) < len(connection_list):
            self.sock.listen(5)
            fed_logger.info("Waiting For Incoming Connections.")
            (edge_sock, (ip, port)) = self.sock.accept()
            fed_logger.info('Got connection from ' + str(ip))
            fed_logger.info(edge_sock)
            self.socks[str(ip)] = edge_sock
        model_len = model_utils.get_unit_model_len()
        self.uninet = model_utils.get_model('Unit', [model_len - 1, model_len - 1], self.device)

        self.testset = data_utils.get_testset()
        self.testloader = data_utils.get_testloader(self.testset, multiprocessing.cpu_count())
        self.criterion = nn.CrossEntropyLoss()

    @abstractmethod
    def offloading_train(self, client_ips):
        pass

    @abstractmethod
    def no_offloading_train(self, client_ips):
        pass

    @abstractmethod
    def test_network(self, edge_ips):
        """
        send message to test_app network speed
        """
        pass

    @abstractmethod
    def client_network(self, edge_ips):
        """
        receive client network speed
        """
        pass

    @abstractmethod
    def split_layer(self):
        """
        send splitting data
        """
        pass

    @abstractmethod
    def e_local_weights(self, client_ips):
        """
        receive final weights for aggregation in offloading mode
        """
        pass

    @abstractmethod
    def c_local_weights(self, client_ips):
        """
        receive client local weights in no offloading mode
        """
        pass

    @abstractmethod
    def offloading_global_weights(self):
        """
        send global weights
        """
        pass

    @abstractmethod
    def no_offloading_global_weights(self):
        pass

    @abstractmethod
    def initialize(self, split_layers, LR):
        pass

    @abstractmethod
    def aggregate(self, client_ips, aggregate_method, eweights):
        pass

    def call_aggregation(self, options: dict, eweights):
        method = fl_method_parser.fl_methods.get(options.get('aggregation'))
        if method is None:
            fed_logger.error("aggregate method is none")
        self.aggregate(config.CLIENTS_LIST, method, eweights)

    @abstractmethod
    def cluster(self, options: dict):
        pass

    @abstractmethod
    def split(self, options: dict):
        pass

    def scatter(self, msg):
        for i in self.socks:
            self.send_msg(self.socks[i], msg)

    def concat_norm(self, ttpi, offloading):
        ttpi_order = []
        offloading_order = []
        for c in config.CLIENTS_LIST:
            ttpi_order.append(ttpi[c])
            offloading_order.append(offloading[c])

        group_max_index = [0 for i in range(config.G)]
        group_max_value = [0 for i in range(config.G)]
        for i in range(len(config.CLIENTS_LIST)):
            label = self.group_labels[i]
            if ttpi_order[i] >= group_max_value[label]:
                group_max_value[label] = ttpi_order[i]
                group_max_index[label] = i

        ttpi_order = np.array(ttpi_order)[np.array(group_max_index)]
        offloading_order = np.array(offloading_order)[np.array(group_max_index)]
        state = np.append(ttpi_order, offloading_order)
        return state

    def get_offloading(self, split_layer):
        offloading = {}
        workload = 0

        assert len(split_layer) == len(config.CLIENTS_LIST)
        for i in range(len(config.CLIENTS_LIST)):
            for l in range(model_utils.get_unit_model_len()):
                if l <= split_layer[i][0]:
                    workload += model_utils.get_class()().cfg[l][5]
            offloading[config.CLIENTS_LIST[i]] = workload / config.total_flops
            workload = 0

        return offloading

    def ttpi(self, client_ips):
        ttpi = {}
        for i in range(len(client_ips)):
            # ttpi[str(client_ips[i])] = self.tt_end[client_ips[i]] - self.tt_start[client_ips[i]]
            ttpi[str(client_ips[i])] = 3
        return ttpi

    def bandwith(self):
        return self.edge_bandwidth
