from abc import ABC, abstractmethod

import numpy as np
import torch

from config import config
from config.logger import fed_logger
from entity.Communicator import Communicator
from fl_method import fl_method_parser
from util import model_utils, message_utils


class FedEdgeServerInterface(ABC, Communicator):
    def __init__(self, index, ip_address, server_port, model_name, dataset):
        super(FedEdgeServerInterface, self).__init__(index, ip_address)
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
        self.threads = None
        self.net_threads = None
        self.ttpi = None
        self.offloading = None
        while len(self.client_socks) < len(config.EDGE_MAP[ip_address]):
            self.sock.listen(5)
            fed_logger.info("Waiting Incoming Connections.")
            (client_sock, (ip, port)) = self.sock.accept()
            fed_logger.info('Got connection from ' + str(ip))
            fed_logger.info(client_sock)
            self.client_socks[str(ip)] = client_sock

        self.uninet = model_utils.get_model('Unit', config.model_len - 1, self.device)

    @abstractmethod
    def initialize(self, split_layers, offload, first, LR):
        pass

    @abstractmethod
    def train(self, thread_number, client_ips):
        pass

    @abstractmethod
    def aggregate(self, client_ips, aggregate_method):
        pass

    def post_train(self, options: dict):
        self.offloading = self.get_offloading(self.split_layers)
        self.cluster(options)
        state = self.concat_norm(self.ttpi, self.offloading)
        return state, self.bandwidth

    def call_aggregation(self, options: dict):
        method = fl_method_parser.fl_methods.get(options.get('aggregation'))
        if method is None:
            fed_logger.error("aggregate method is none")
        self.aggregate(config.CLIENTS_LIST, method)

    def cluster(self, options: dict):
        self.group_labels = fl_method_parser.fl_methods.get(options.get('clustering'))()

    # def split(self, options: dict):
    #     self.split_layers = fl_method_parser.fl_methods.get(options.get('splitting'))(self.state, self.group_labels)
    #     fed_logger.info('Next Round OPs: ' + str(config.split_layer))
    #     msg = [message_utils.split_layers, config.split_layer]
    #     self.scatter(msg)

    def scatter(self, msg):
        for i in self.client_socks:
            self.send_msg(self.client_socks[i], msg)

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
            for l in range(len(config.model_cfg[config.model_name])):
                if l <= split_layer[i]:
                    workload += config.model_cfg[config.model_name][l][5]
            offloading[config.CLIENTS_LIST[i]] = workload / config.total_flops
            workload = 0

        return offloading
