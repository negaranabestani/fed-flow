import multiprocessing
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn

from app.config import config
from app.config.logger import fed_logger
from app.entity.communicator import Communicator
from app.entity.node import Node
from app.fl_method import fl_method_parser
from app.util import data_utils, model_utils


class FedServerInterface(Node, ABC, Communicator):
    def __init__(self, ip: str, port: int, model_name, dataset, offload, edge_based):
        Node.__init__(self, ip, port)
        Communicator.__init__(self)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.offload = offload
        self.edge_based = edge_based
        self.model_name = model_name
        self.group_labels = None
        self.criterion = None
        self.split_layers = None
        # self.state = None
        self.client_bandwidth = {}
        self.edge_bandwidth = {}
        self.dataset = dataset
        self.threads = None
        self.net_threads = None
        self.offloading = None
        self.tt_start = {}
        self.tt_end = {}

        self.uninet = model_utils.get_model('Unit', None, self.device, self.edge_based)

        self.testset = data_utils.get_testset()
        self.testloader = data_utils.get_testloader(self.testset, multiprocessing.cpu_count())
        self.criterion = nn.CrossEntropyLoss()

    @abstractmethod
    def edge_offloading_train(self, client_ips):
        pass

    @abstractmethod
    def no_edge_offloading_train(self, client_ips):
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
    def get_split_layers_config_from_edge(self):
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
    def edge_offloading_global_weights(self):
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
    def split(self, state, options: dict):
        pass

    def scatter(self, msg, is_weight=False):
        list1 = config.CLIENTS_LIST
        if self.edge_based:
            list1 = config.EDGE_SERVER_LIST
            for i in list1:
                self.send_msg(exchange=i, msg=msg, is_weight=is_weight, url=i)

        else:
            for i in list1:
                self.send_msg(exchange=i, msg=msg, is_weight=is_weight)

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
                split_point = split_layer[i]
                if self.edge_based:
                    split_point = split_layer[i][0]
                if l <= split_point:
                    workload += model_utils.get_class()().cfg[l][5]
            offloading[config.CLIENTS_LIST[i]] = workload / config.total_flops
            workload = 0

        return offloading

    def ttpi(self, client_ips):
        ttpi = {}
        for i in range(len(client_ips)):
            # ttpi[str(client_ips[i])] = self.tt_end[client_ips[i]] - self.tt_start[client_ips[i]]
            ttpi[config.CLIENTS_LIST[i]] = 3
        return ttpi

    def bandwith(self):
        return self.edge_bandwidth

    @abstractmethod
    def e_energy_tt(self, client_ips):
        pass

    @abstractmethod
    def edge_based_state(self):
        pass
