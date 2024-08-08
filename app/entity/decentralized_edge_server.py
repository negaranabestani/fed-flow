import threading
import time

import torch
from torch import optim, nn, multiprocessing

from app.config import config
from app.config.logger import fed_logger
from app.dto.message import NetworkTestMessage, IterationFlagMessage, GlobalWeightMessage
from app.entity.communicator import Communicator
from app.entity.fed_base_node_interface import FedBaseNodeInterface
from app.entity.http_communicator import HTTPCommunicator
from app.entity.node import NodeType, NodeIdentifier, Node
from app.fl_method import fl_method_parser
from app.util import model_utils, data_utils


# noinspection PyTypeChecker
class FedDecentralizedEdgeServer(FedBaseNodeInterface):

    def __init__(self, ip: str, port: int, model_name, dataset, offload):
        Node.__init__(self, ip, port, NodeType.EDGE)
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
        self.offload = offload

        if offload:
            model_len = model_utils.get_unit_model_len()
            self.uninet = model_utils.get_model('Unit', [model_len - 1, model_len - 1], self.device, True)

            self.testset = data_utils.get_testset()
            self.testloader = data_utils.get_testloader(self.testset, multiprocessing.cpu_count())
        self.neighbor_bandwidth = {}
        self.optimizers = None
        self.nets = None
        self.split_layers = {}

    def initialize(self, learning_rate, **kwargs):
        self.nets = {}
        self.optimizers = {}
        for neighbor in self.get_neighbors():
            neighbor_type = HTTPCommunicator.get_node_type(neighbor)
            if neighbor_type != NodeType.CLIENT:
                continue
            client_id = str(neighbor)
            if client_id not in self.split_layers:
                self.split_layers[client_id] = len(self.uninet.cfg) - 2
            split_point = self.split_layers[client_id]
            if split_point < len(self.uninet.cfg) - 1:
                self.nets[client_id] = model_utils.get_model('Edge', split_point, self.device, False)
                cweights = model_utils.get_model('Client', split_point, self.device, False).state_dict()
                pweights = model_utils.split_weights_server(self.uninet.state_dict(), cweights,
                                                            self.nets[client_id].state_dict(), [])
                self.nets[client_id].load_state_dict(pweights)

                if len(list(self.nets[client_id].parameters())) != 0:
                    self.optimizers[client_id] = optim.SGD(self.nets[client_id].parameters(), lr=learning_rate,
                                                           momentum=0.9)
            else:
                self.nets[client_id] = model_utils.get_model('Edge', split_point, self.device, False)
        self.criterion = nn.CrossEntropyLoss()

    def gather_neighbors_network_bandwidth(self, neighbors_type: NodeType = None):
        net_threads = {}
        for neighbor in self.get_neighbors():
            neighbor_type = HTTPCommunicator.get_node_type(neighbor)
            if neighbors_type is None or neighbor_type == neighbors_type:
                net_threads[str(neighbor)] = threading.Thread(target=self._thread_network_testing,
                                                              args=(neighbor,), name=str(neighbor))
                net_threads[str(neighbor)].start()

        for _, thread in net_threads.items():
            thread.join()

    def _thread_network_testing(self, neighbor: NodeIdentifier):
        network_time_start = time.time()
        msg = NetworkTestMessage([self.uninet.cpu().state_dict()])
        neighbor_rabbitmq_url = HTTPCommunicator.get_rabbitmq_url(neighbor)
        self.send_msg(self.get_exchange_name(), neighbor_rabbitmq_url, msg)
        fed_logger.info("server test network sent")
        msg: NetworkTestMessage = self.recv_msg(neighbor.get_exchange_name(), config.current_node_mq_url,
                                                NetworkTestMessage.MESSAGE_TYPE)
        fed_logger.info("server test network received")
        network_time_end = time.time()
        self.neighbor_bandwidth[str(neighbor)] = data_utils.sizeofmessage(msg.weights) / (
                network_time_end - network_time_start)

    def cluster(self, options: dict):
        self.group_labels = fl_method_parser.fl_methods.get(options.get('clustering'))()

    def get_state(self):
        state = []
        for neighbor in self.get_neighbors():
            state.append(self.neighbor_bandwidth[str(neighbor)])
        return state

    def split(self, state, options: dict):
        self.split_layers = fl_method_parser.fl_methods.get(options.get('splitting'))(state, self.group_labels, self)
        fed_logger.info('Next Round OPs: ' + str(self.split_layers))

    def start_offloading_train(self):
        self.threads = {}
        for neighbor in self.get_neighbors():
            self.threads[str(neighbor)] = threading.Thread(target=self._thread_offload_training,
                                                           args=(neighbor,), name=str(neighbor))
            fed_logger.info(str(neighbor) + ' offloading training start')
            self.threads[str(neighbor)].start()

        for neighbor in self.get_neighbors():
            self.threads[str(neighbor)].join()

    def _thread_offload_training(self, neighbor: NodeIdentifier):
        neighbor_rabbitmq_url = HTTPCommunicator.get_rabbitmq_url(neighbor)
        flag: bool = self.recv_msg(neighbor.get_exchange_name(), neighbor_rabbitmq_url,
                                   IterationFlagMessage.MESSAGE_TYPE).flag
        while flag:
            flag = self.recv_msg(neighbor.get_exchange_name(), neighbor_rabbitmq_url,
                                 IterationFlagMessage.MESSAGE_TYPE).flag
            if not flag:
                break
            msg: GlobalWeightMessage = self.recv_msg(neighbor.get_exchange_name(), neighbor_rabbitmq_url,
                                                     GlobalWeightMessage.MESSAGE_TYPE)
            smashed_layers = msg.weights[0]
            labels = msg.weights[1]
            inputs, targets = smashed_layers.to(self.device), labels.to(self.device)
            if self.split_layers[str(neighbor)] < len(self.uninet.cfg) - 1:
                if str(neighbor) in self.optimizers.keys():
                    self.optimizers[str(neighbor)].zero_grad()
            outputs = self.nets[str(neighbor)](inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            if self.split_layers[str(neighbor)] < len(self.uninet.cfg) - 1:
                if str(neighbor) in self.optimizers.keys():
                    self.optimizers[str(neighbor)].step()

            fed_logger.info(str(neighbor) + " sending gradients")
            msg = GlobalWeightMessage([inputs.grad])
            self.send_msg(self.get_exchange_name(), config.mq_url, msg)

        fed_logger.info(str(neighbor) + ' offloading training end')

    def gather_local_weights(self) -> dict[any]:
        eweights = {}
        for neighbor in self.get_neighbors():
            neighbor_type = HTTPCommunicator.get_node_type(neighbor)
            if neighbor_type == NodeType.CLIENT:
                neighbor_rabbitmq_url = HTTPCommunicator.get_rabbitmq_url(neighbor)
                msg: GlobalWeightMessage = self.recv_msg(neighbor.get_exchange_name(), neighbor_rabbitmq_url,
                                                         GlobalWeightMessage.MESSAGE_TYPE)
                eweights[str(neighbor)] = msg.weights[0]
        return eweights

    def call_aggregation(self, options: dict, eweights):
        method = fl_method_parser.fl_methods.get(options.get('aggregation'))
        if method is None:
            fed_logger.error("aggregate method is none")
        self.aggregate(method, eweights)

    def aggregate(self, aggregate_method, eweights):
        w_local_list = []
        for neighbor in self.get_neighbors():
            if self.offload:
                split_point = self.split_layers[str(neighbor)]
                if split_point != (config.model_len - 1):
                    w_local = (
                        model_utils.concat_weights(self.uninet.state_dict(), eweights[str(neighbor)],
                                                   self.nets[str(neighbor)].state_dict()),
                        config.N / config.K)
                    w_local_list.append(w_local)
                else:
                    w_local = (eweights[str(neighbor)], config.N / config.K)
            else:
                w_local = (eweights[str(neighbor)], config.N / config.K)
            w_local_list.append(w_local)
        zero_model = model_utils.zero_init(self.uninet).state_dict()
        aggregated_model = aggregate_method(zero_model, w_local_list, config.N)
        self.uninet.load_state_dict(aggregated_model)
        return aggregated_model

    def bandwidth(self) -> dict[str, float]:
        return self.neighbor_bandwidth
