import threading
import time
from abc import ABC, abstractmethod

import torch
from torch import multiprocessing

from config import config
from config.logger import fed_logger
from entity.Communicator import Communicator
from util import model_utils, message_utils, data_utils


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

    def test_client_network(self, client_ips):
        """
        send message to test network speed
        """
        # Network test
        self.net_threads = {}
        for i in range(len(client_ips)):
            self.net_threads[client_ips[i]] = threading.Thread(target=self._thread_client_network_testing,
                                                               args=(client_ips[i],))
            self.net_threads[client_ips[i]].start()

        for i in range(len(client_ips)):
            self.net_threads[client_ips[i]].join()

    def _thread_client_network_testing(self, client_ip):
        network_time_start = time.time()
        msg = [message_utils.test_client_network, self.uninet.cpu().state_dict()]
        self.send_msg(self.socks[client_ip], msg)
        msg = self.recv_msg(self.socks[client_ip], message_utils.test_client_network)
        network_time_end = time.time()
        self.client_bandwidth[client_ip] = network_time_end - network_time_start

    def test_server_network(self):
        msg = self.recv_msg(self.sock, message_utils.test_server_network)
        msg = [message_utils.test_server_network, self.uninet.cpu().state_dict()]
        self.send_msg(self.sock, msg)

    def client_network(self):
        """
        send client network speed to central server
        """
        msg = [message_utils.client_network, self.client_bandwidth]
        self.send_msg(self.sock, msg)

    def split_layer(self):
        """
        receive send splitting data to clients
        """
        msg = self.recv_msg(self.sock, message_utils.split_layers_server_to_edge)
        self.split_layers = msg[1]
        msg = [message_utils.split_layers_edge_to_client, self.split_layers]
        self.scatter(msg)

    def local_weights(self, client_ip):
        """
        receive and send final weights for aggregation
        """
        cweights = self.recv_msg(self.socks[client_ip],
                                 message_utils.local_weights_edge_to_server)[1]

        msg = [message_utils.local_weights_edge_to_server, cweights]
        self.send_msg(self.sock, msg)

    def global_weights(self, client_ips: []):
        """
        receive global weights
        """
        weights = self.recv_msg(self.sock, message_utils.local_weights_edge_to_server)[1]
        for i in range(len(self.split_layers)):
            if vars(client_ips).__contains__(config.CLIENTS_LIST[i]):
                cweights = model_utils.get_model('Client', self.split_layers[i], self.device).state_dict()

                pweights = model_utils.split_weights_edgeserver(weights, cweights,
                                                                self.nets[config.CLIENTS_LIST[i]].state_dict())
                self.nets[config.CLIENTS_LIST[i]].load_state_dict(pweights)

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
        outputs = self.forward_propagation(client_ip)
        self.backward_propagation(outputs, client_ip)
