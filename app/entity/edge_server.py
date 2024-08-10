import threading
import time

import torch
from colorama import Fore
from torch import optim, nn, multiprocessing

from app.config import config
from app.config.logger import fed_logger
from app.dto.message import JsonMessage, GlobalWeightMessage, NetworkTestMessage, IterationFlagMessage, \
    SplitLayerConfigMessage
from app.entity.communicator import Communicator
from app.entity.fed_base_node_interface import FedBaseNodeInterface
from app.entity.node import Node
from app.entity.node_type import NodeType
from app.util import model_utils, data_utils


# noinspection PyTypeChecker
class FedEdgeServer(FedBaseNodeInterface):
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

    def initialize(self, split_layers, LR, client_ips):
        self.split_layers = split_layers
        self.optimizers = {}
        for i in range(len(split_layers)):
            if config.CLIENTS_LIST[i] in client_ips:
                client_ip = config.CLIENTS_LIST[i]
                if split_layers[i][0] < split_layers[i][
                    1]:  # Only offloading client need initialize optimizer in server
                    self.nets[client_ip] = model_utils.get_model('Edge', split_layers[i], self.device, True)

                    # offloading weight in server also need to be initialized from the same global weight
                    cweights = model_utils.get_model('Client', split_layers[i], self.device, True).state_dict()

                    pweights = model_utils.split_weights_edgeserver(self.uninet.state_dict(), cweights,
                                                                    self.nets[client_ip].state_dict())
                    self.nets[client_ip].load_state_dict(pweights)
                    if len(list(self.nets[client_ip].parameters())) != 0:
                        self.optimizers[client_ip] = optim.SGD(self.nets[client_ip].parameters(), lr=LR,
                                                               momentum=0.9)
                else:
                    self.nets[client_ip] = model_utils.get_model('Edge', split_layers[i], self.device, True)
        self.criterion = nn.CrossEntropyLoss()

    def aggregate(self, client_ips, aggregate_method):
        pass

    def forward_propagation(self, client_ip):
        i = 0
        edge_exchange = 'edge.' + client_ip
        server_exchange = 'server.' + client_ip
        client_exchange = 'client.' + client_ip
        msg: IterationFlagMessage = self.recv_msg(edge_exchange, config.mq_url, IterationFlagMessage.MESSAGE_TYPE)
        flag: bool = msg.flag
        if self.split_layers[config.CLIENTS_NAME_TO_INDEX.get(client_ip)][1] < model_utils.get_unit_model_len() - 1:
            self.send_msg(server_exchange, config.mq_url, IterationFlagMessage(flag))
        else:
            self.send_msg(server_exchange, config.mq_url, IterationFlagMessage(False))
        i += 1

        while flag:
            if self.split_layers[config.CLIENTS_NAME_TO_INDEX.get(client_ip)][0] < model_utils.get_unit_model_len() - 1:
                msg: IterationFlagMessage = self.recv_msg(edge_exchange, config.mq_url,
                                                          IterationFlagMessage.MESSAGE_TYPE)
                flag: bool = msg.flag

                if not flag:
                    self.send_msg(server_exchange, config.mq_url, IterationFlagMessage(flag))
                    break
                # fed_logger.info(client_ip + " receiving local activations")
                msg: GlobalWeightMessage = self.recv_msg(edge_exchange,
                                                         config.mq_url,
                                                         GlobalWeightMessage.MESSAGE_TYPE)
                smashed_layers = msg.weights[0]
                labels = msg.weights[1]

                # fed_logger.info(client_ip + " training model forward")

                inputs, targets = smashed_layers.to(self.device), labels.to(self.device)
                if self.split_layers[config.CLIENTS_NAME_TO_INDEX[client_ip]][0] < \
                        self.split_layers[config.CLIENTS_NAME_TO_INDEX[client_ip]][1]:
                    if self.optimizers.keys().__contains__(client_ip):
                        self.optimizers[client_ip].zero_grad()
                    outputs = self.nets[client_ip](inputs)
                    # fed_logger.info(client_ip + " sending local activations")
                    if self.split_layers[config.CLIENTS_NAME_TO_INDEX[client_ip]][
                        1] < model_utils.get_unit_model_len() - 1:
                        self.send_msg(server_exchange, config.mq_url, IterationFlagMessage(flag))
                        msg: list = [outputs.cpu(), targets.cpu()]
                        self.send_msg(server_exchange, config.mq_url, GlobalWeightMessage(msg))
                        msg: GlobalWeightMessage = self.recv_msg(edge_exchange,
                                                                 config.mq_url,
                                                                 GlobalWeightMessage.MESSAGE_TYPE)
                        gradients = msg.weights[0].to(self.device)
                        # fed_logger.info(client_ip + " training model backward")
                        outputs.backward(gradients)
                        msg: list = [inputs.grad]
                        self.send_msg(client_exchange, config.mq_url, GlobalWeightMessage(msg))
                    else:
                        outputs = self.nets[client_ip](inputs)
                        loss = self.criterion(outputs, targets)
                        loss.backward()
                        if client_ip in self.optimizers.keys():
                            self.optimizers[client_ip].step()
                        msg: list = [inputs.grad]
                        self.send_msg(client_exchange, config.mq_url, GlobalWeightMessage(msg))
                else:
                    self.send_msg(server_exchange, config.mq_url, IterationFlagMessage(flag))
                    msg: list = [inputs.cpu(), targets.cpu()]
                    self.send_msg(server_exchange, config.mq_url, GlobalWeightMessage(msg))
                    # fed_logger.info(client_ip + " edge receiving gradients")
                    msg: GlobalWeightMessage = self.recv_msg(edge_exchange,
                                                             config.mq_url,
                                                             GlobalWeightMessage.MESSAGE_TYPE)
                    # fed_logger.info(client_ip + " edge received gradients")
                    self.send_msg(client_exchange, config.mq_url, msg)
            i += 1
        fed_logger.info(str(client_ip) + ' offloading training end')

    def backward_propagation(self, outputs, client_ip, inputs):
        pass

    def test_client_network(self, client_ips):
        """
        send message to test_app network speed
        """
        # Network test_app
        self.net_threads = {}
        for i in range(len(client_ips)):
            self.net_threads[client_ips[i]] = threading.Thread(target=self._thread_client_network_testing,
                                                               args=(client_ips[i],))
            self.net_threads[client_ips[i]].start()

        for i in range(len(client_ips)):
            self.net_threads[client_ips[i]].join()

    def _thread_client_network_testing(self, client_ip):
        network_time_start = time.time()
        msg: list = [self.uninet.cpu().state_dict()]
        self.send_msg(client_ip, config.mq_url, NetworkTestMessage(msg))
        msg: NetworkTestMessage = self.recv_msg(config.EDGE_SERVER_INDEX_TO_NAME[config.index], config.mq_url,
                                                NetworkTestMessage.MESSAGE_TYPE)
        network_time_end = time.time()
        self.client_bandwidth[client_ip] = data_utils.sizeofmessage(msg.weights) / (
                network_time_end - network_time_start)

    def test_server_network(self):
        _ = self.recv_msg(config.EDGE_SERVER_INDEX_TO_NAME[config.index],
                          config.mq_url, NetworkTestMessage.MESSAGE_TYPE)
        server_exchange = 'server.' + config.EDGE_SERVER_INDEX_TO_NAME[config.index]
        msg: list = [self.uninet.cpu().state_dict()]
        self.send_msg(server_exchange, config.mq_url, NetworkTestMessage(msg))

    def client_network(self):
        """
        send client network speed to central server
        """
        server_exchange = 'server.' + config.EDGE_SERVER_INDEX_TO_NAME[config.index]
        msg = [self.client_bandwidth]
        self.send_msg(server_exchange, config.mq_url, JsonMessage(msg))

    def get_split_layers_config(self, client_ips):
        """
        receive send splitting data to clients
        """
        msg: SplitLayerConfigMessage = self.recv_msg(config.EDGE_SERVER_INDEX_TO_NAME[config.index], config.mq_url,
                                                     SplitLayerConfigMessage.MESSAGE_TYPE)
        self.split_layers = msg.data
        fed_logger.info(Fore.LIGHTYELLOW_EX + f"{msg.data}" + Fore.RESET)
        msg = SplitLayerConfigMessage(self.split_layers)
        self.scatter(msg)

    def local_weights(self, client_ip):
        """
        receive and send final weights for aggregation
        """
        edge_exchange = 'edge.' + client_ip
        server_exchange = 'server.' + client_ip
        cweights = self.recv_msg(edge_exchange, config.mq_url, GlobalWeightMessage.MESSAGE_TYPE).weights[0]
        sp = self.split_layers[config.CLIENTS_NAME_TO_INDEX[client_ip]][0]
        if sp != (config.model_len - 1):
            w_local = model_utils.concat_weights(self.uninet.state_dict(), cweights,
                                                 self.nets[client_ip].state_dict())
        else:
            w_local = cweights
            # print("------------------------"+str(eweights[i]))
        msg = GlobalWeightMessage([w_local])
        self.send_msg(server_exchange, config.mq_url, msg)

    def energy(self, client_ips):
        energy_tt_list = []
        for client_ip in client_ips:
            ms: JsonMessage = self.recv_msg(client_ip, config.mq_url, JsonMessage.MESSAGE_TYPE)
            energy_tt_list.append([ms.data[0], ms.data[1]])
        # fed_logger.info(f"sending enery tt {socket.gethostname()}")
        msg = energy_tt_list
        self.send_msg(config.EDGE_SERVER_INDEX_TO_NAME[config.index], config.mq_url, JsonMessage(msg))

    def get_global_weights(self, client_ips: []):
        """
        receive and send global weights
        """
        msg: GlobalWeightMessage = self.recv_msg(config.EDGE_SERVER_INDEX_TO_NAME[config.index],
                                                 config.mq_url, GlobalWeightMessage.MESSAGE_TYPE)
        weights = msg.weights[0]
        for i in range(len(self.split_layers)):
            if config.CLIENTS_LIST[i] in client_ips:
                cweights = model_utils.get_model('Client', self.split_layers[i], self.device, True).state_dict()
                pweights = model_utils.split_weights_edgeserver(weights, cweights,
                                                                self.nets[config.CLIENTS_LIST[i]].state_dict())
                self.nets[config.CLIENTS_LIST[i]].load_state_dict(pweights)
        self.scatter(GlobalWeightMessage([weights]))

    def no_offload_global_weights(self):
        msg: GlobalWeightMessage = \
            self.recv_msg(config.EDGE_SERVER_INDEX_TO_NAME[config.index],
                          config.mq_url, GlobalWeightMessage.MESSAGE_TYPE)
        weights = msg.weights[0]
        self.scatter(GlobalWeightMessage([weights]))

    def thread_offload_training(self, client_ip):
        self.forward_propagation(client_ip)
        self.local_weights(client_ip)
        # self.energy(client_ip)

    def thread_no_offload_training(self, client_ip):
        self.local_weights(client_ip)
