import multiprocessing
import threading
import time
from abc import ABC, abstractmethod

import numpy as np
import torch

from config import config
from config.logger import fed_logger
from entity.Communicator import Communicator
from fl_method import fl_method_parser
from util import model_utils, data_utils, message_utils


class FedServerInterface(ABC, Communicator):
    def __init__(self, index, ip_address, server_port, model_name, dataset):
        super(FedServerInterface, self).__init__(index, ip_address)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.port = server_port
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
        self.ttpi = None
        self.offloading = None
        self.tt_start = {}
        self.tt_end = {}

        while len(self.socks) < config.K:
            self.sock.listen(5)
            fed_logger.info("Waiting Incoming Connections.")
            (edge_sock, (ip, port)) = self.sock.accept()
            fed_logger.info('Got connection from ' + str(ip))
            fed_logger.info(edge_sock)
            self.socks[str(ip)] = edge_sock

        self.uninet = model_utils.get_model('Unit', config.model_len - 1, self.device)

        self.testset = data_utils.get_testset()
        self.testloader = data_utils.get_testloader(self.testset, multiprocessing.cpu_count())

    @abstractmethod
    def offloading_train(self, client_ips):
        pass

    @abstractmethod
    def no_offloading_train(self, client_ips):
        pass

    def test_network(self, edge_ips):
        """
        send message to test network speed
        """
        # Network test
        self.net_threads = {}
        for i in range(len(edge_ips)):
            self.net_threads[edge_ips[i]] = threading.Thread(target=self._thread_network_testing,
                                                             args=(edge_ips[i],))
            self.net_threads[edge_ips[i]].start()

        for i in range(len(edge_ips)):
            self.net_threads[edge_ips[i]].join()

    def _thread_network_testing(self, edge_ip):
        network_time_start = time.time()
        msg = [message_utils.test_network, self.uninet.cpu().state_dict()]
        self.send_msg(self.socks[edge_ip], msg)
        msg = self.recv_msg(self.socks[edge_ip], message_utils.test_network)
        network_time_end = time.time()
        self.edge_bandwidth[edge_ip] = network_time_end - network_time_start

    def client_network(self, edge_ips):
        """
        receive client network speed
        """
        for i in edge_ips:
            msg = self.recv_msg(self.socks[edge_ips[i]], message_utils.client_network)
            self.client_bandwidth[i] = msg

    def split_layer(self):
        """
        send splitting data
        """
        msg = [message_utils.split_layers, config.split_layer]
        self.scatter(msg)

    def e_local_weights(self, client_ips):
        """
        send final weights for aggregation
        """
        eweights = []
        for i in range(len(client_ips)):
            msg = self.recv_msg(self.socks[config.CLIENT_MAP[client_ips[i]]],
                                message_utils.local_weights_client_to_server)
            self.tt_end[client_ips[i]] = time.time()
            eweights.append(msg)
        return eweights

    def c_local_weights(self, client_ips):
        cweights = []
        for i in range(len(client_ips)):
            msg = self.recv_msg(self.socks[client_ips[i]],
                                message_utils.local_weights_client_to_server)
            self.tt_end[client_ips[i]] = time.time()
            cweights.append(msg)
        return cweights

    def global_weights(self):
        """
        send global weights
        """
        msg = [message_utils.initial_global_weights_server_to_client, self.uninet.state_dict()]
        self.scatter(msg)

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

    def cluster(self, options: dict):
        self.group_labels = fl_method_parser.fl_methods.get(options.get('clustering'))()

    def split(self, options: dict):
        self.split_layers = fl_method_parser.fl_methods.get(options.get('splitting'))(self.state, self.group_labels)
        fed_logger.info('Next Round OPs: ' + str(self.split_layer))

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
            for l in range(len(config.model_cfg[config.model_name])):
                if l <= split_layer[i]:
                    workload += config.model_cfg[config.model_name][l][5]
            offloading[config.CLIENTS_LIST[i]] = workload / config.total_flops
            workload = 0

        return offloading

    def ttpi(self, client_ips):
        ttpi = {}
        for i in range(len(client_ips)):
            ttpi[client_ips[i]] = self.tt_end[client_ips[i]] - self.tt_start[client_ips[i]]
        return ttpi

    def bandwith(self):
        return self.edge_bandwidth
