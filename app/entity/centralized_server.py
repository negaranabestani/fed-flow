import threading
import time

import numpy as np
from colorama import Fore
from torch import nn, optim

from app.config import config
from app.config.logger import fed_logger
from app.dto.bandwidth import BandWidth
from app.dto.message import IterationFlagMessage, GlobalWeightMessage, NetworkTestMessage, JsonMessage, \
    SplitLayerConfigMessage, BaseMessage
from app.entity.fed_base_node_interface import FedBaseNodeInterface
from app.entity.http_communicator import HTTPCommunicator
from app.entity.node_identifier import NodeIdentifier
from app.entity.node_type import NodeType
from app.fl_method import fl_method_parser
from app.model.utils import get_available_torch_device
from app.util import model_utils, data_utils


# noinspection PyTypeChecker
class CentralizedServer(FedBaseNodeInterface):
    def __init__(self, ip: str, port: int, model_name, dataset, offload, edge_based):
        super().__init__(ip, port, NodeType.SERVER)

        self.optimizers = {}
        self.nets = {}
        self.device = get_available_torch_device()
        self.offload = offload
        self.edge_based = edge_based
        self.model_name = model_name
        self.group_labels = None
        self.criterion = None
        self.split_layers = None
        self.neighbors_bandwidth: dict[str, BandWidth] = {}
        self.dataset = dataset
        self.threads = None
        self.net_threads = None
        self.offloading = None
        self.tt_start = {}
        self.tt_end = {}

        self.uninet = model_utils.get_model('Unit', None, self.device, self.edge_based)

        self.testset = data_utils.get_testset()
        self.testloader = data_utils.get_testloader(self.testset, 0)
        self.criterion = nn.CrossEntropyLoss()

    def initialize(self, split_layers, LR):
        self.split_layers = split_layers
        self.nets = {}
        self.optimizers = {}
        for i in range(len(split_layers)):
            client_ip = config.CLIENTS_LIST[i]
            split_point = split_layers[i]
            if self.edge_based:
                split_point = split_layers[i][1]
            if split_point < len(
                    self.uninet.cfg) - 1:  # Only offloading client need initialize optimizer in server
                if self.edge_based:
                    self.nets[client_ip] = model_utils.get_model('Server', split_layers[i], self.device,
                                                                 self.edge_based)

                    # offloading weight in server also need to be initialized from the same global weight
                    eweights = model_utils.get_model('Edge', split_layers[i], self.device, self.edge_based).state_dict()
                    cweights = model_utils.get_model('Client', split_layers[i], self.device,
                                                     self.edge_based).state_dict()

                    pweights = model_utils.split_weights_server(self.uninet.state_dict(), cweights,
                                                                self.nets[client_ip].state_dict(), eweights)
                    self.nets[client_ip].load_state_dict(pweights)

                    if len(list(self.nets[client_ip].parameters())) != 0:
                        self.optimizers[client_ip] = optim.SGD(self.nets[client_ip].parameters(), lr=LR,
                                                               momentum=0.9)
                else:
                    self.nets[client_ip] = model_utils.get_model('Server', split_layers[i], self.device,
                                                                 self.edge_based)

                    # offloading weight in server also need to be initialized from the same global weight
                    cweights = model_utils.get_model('Client', split_layers[i], self.device,
                                                     self.edge_based).state_dict()
                    pweights = model_utils.split_weights_server(self.uninet.state_dict(), cweights,
                                                                self.nets[client_ip].state_dict(), [])
                    self.nets[client_ip].load_state_dict(pweights)

                    if len(list(self.nets[client_ip].parameters())) != 0:
                        self.optimizers[client_ip] = optim.SGD(self.nets[client_ip].parameters(), lr=LR,
                                                               momentum=0.9)
            else:
                self.nets[client_ip] = model_utils.get_model('Server', split_layers[i], self.device, self.edge_based)
        self.criterion = nn.CrossEntropyLoss()

    def start_edge_training(self):
        clients_by_edge = self._get_clients_by_edge()
        self.threads = {}
        for edge, clients in clients_by_edge.items():
            for client in clients:
                client_key = f'{edge}_{client}'
                self.threads[client_key] = threading.Thread(target=self._thread_edge_training,
                                                            args=(edge, client))
                self.threads[client_key].start()
                self.tt_start[client_key] = time.time()

        for thread in self.threads:
            self.threads[thread].join()

    def _get_clients_by_edge(self) -> dict[NodeIdentifier, list[NodeIdentifier]]:
        clients_by_edge = {}
        edge_neighbors = self.get_neighbors([NodeType.EDGE])
        for edge in edge_neighbors:
            clients_by_edge[edge] = HTTPCommunicator.get_neighbors_from_neighbor(edge, [NodeType.CLIENT])
        return clients_by_edge

    def _thread_edge_training(self, edge: NodeIdentifier, client: NodeIdentifier):
        edge_exchange_name = edge.get_exchange_name(client)
        flag: bool = self.recv_msg(edge_exchange_name, config.current_node_mq_url,
                                   IterationFlagMessage.MESSAGE_TYPE).flag
        fed_logger.info(Fore.RED + f"{flag}" + Fore.RESET)
        if not flag:
            fed_logger.info(str(client) + ' offloading training end')
            return 'Finish'
        while flag:
            if self.split_layers[client][1] < len(self.uninet.cfg) - 1:
                flag: bool = self.recv_msg(server_exchange, config.mq_url, IterationFlagMessage.MESSAGE_TYPE).flag
                fed_logger.info(Fore.RED + f"{flag}" + Fore.RESET)
                if not flag:
                    break
                msg: GlobalWeightMessage = self.recv_msg(server_exchange, config.mq_url,
                                                         GlobalWeightMessage.MESSAGE_TYPE)
                smashed_layers = msg.weights[0]
                labels = msg.weights[1]
                # fed_logger.info(client_ip + " training model")
                inputs, targets = smashed_layers.to(self.device), labels.to(self.device)
                if self.optimizers.keys().__contains__(client_ip):
                    self.optimizers[client_ip].zero_grad()
                outputs = self.nets[client_ip](inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                if self.optimizers.keys().__contains__(client_ip):
                    self.optimizers[client_ip].step()
                # Send gradients to edge
                # fed_logger.info(client_ip + " sending gradients")
                msg = GlobalWeightMessage([inputs.grad])
                self.send_msg(edge_exchange, config.mq_url, msg)

        fed_logger.info(str(client_ip) + ' offloading training end')
        return 'Finish'

    def aggregate(self, client_ips, aggregate_method, eweights):
        w_local_list = []
        # fed_logger.info("aggregation start")
        for i in range(len(eweights)):
            if self.offload:
                sp = self.split_layers[i]
                if self.edge_based:
                    sp = self.split_layers[i][0]
                if sp != (config.model_len - 1):
                    w_local = (
                        model_utils.concat_weights(self.uninet.state_dict(), eweights[i],
                                                   self.nets[client_ips[i]].state_dict()),
                        config.N / config.K)
                    w_local_list.append(w_local)
                else:
                    w_local = (eweights[i], config.N / config.K)
            else:
                w_local = (eweights[i], config.N / config.K)
            w_local_list.append(w_local)
        zero_model = model_utils.zero_init(self.uninet).state_dict()
        aggregated_model = aggregate_method(zero_model, w_local_list, config.N)
        self.uninet.load_state_dict(aggregated_model)
        return aggregated_model

    def e_local_weights(self, client_ips):
        """
        send final weights for aggregation
        """
        eweights = []
        for i in range(len(client_ips)):
            server_exchange = 'server.' + client_ips[i]
            msg: GlobalWeightMessage = self.recv_msg(server_exchange, config.mq_url,
                                                     GlobalWeightMessage.MESSAGE_TYPE)
            self.tt_end[client_ips[i]] = time.time()
            eweights.append(msg.weights[0])
        return eweights

    def split(self, state, options: dict):
        self.split_layers = fl_method_parser.fl_methods.get(options.get('splitting'))(state, self.group_labels)
        fed_logger.info('Next Round OPs: ' + str(self.split_layers))

    def call_aggregation(self, options: dict, eweights):
        method = fl_method_parser.fl_methods.get(options.get('aggregation'))
        if method is None:
            fed_logger.error("aggregate method is none")
        self.aggregate(config.CLIENTS_LIST, method, eweights)

    def get_neighbors_bandwidth(self) -> dict[str, BandWidth]:
        return self.neighbors_bandwidth
