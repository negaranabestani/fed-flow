import random
import sys
import threading
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from colorama import Fore
from torch import multiprocessing

from app.dto.base_model import BaseModel
from app.dto.message import JsonMessage, GlobalWeightMessage, NetworkTestMessage, SplitLayerConfigMessage, \
    IterationFlagMessage, BaseMessage
from app.entity.aggregators.base_aggregator import BaseAggregator
from app.entity.communicator import Communicator
from app.entity.fed_base_node_interface import FedBaseNodeInterface
from app.entity.http_communicator import HTTPCommunicator
from app.entity.node import Node
from app.entity.node_type import NodeType
from app.fl_method import fl_method_parser

sys.path.append('../../')
from app.util import model_utils, data_utils
from app.config import config
from app.config.logger import fed_logger

np.random.seed(0)
torch.manual_seed(0)


# noinspection PyTypeChecker
class FedServer(FedBaseNodeInterface):
    def __init__(self, ip: str, port: int, model_name, dataset, offload, edge_based, aggregator: BaseAggregator):
        Node.__init__(self, ip, port, NodeType.SERVER, None)
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
        self.aggregator = aggregator

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

    def edge_offloading_train(self, client_ips):
        self.threads = {}
        for i in range(len(client_ips)):
            self.threads[client_ips[i]] = threading.Thread(target=self._thread_edge_training,
                                                           args=(client_ips[i],), name=client_ips[i])
            fed_logger.info(str(client_ips[i]) + ' offloading training start')
            self.threads[client_ips[i]].start()
            self.tt_start[client_ips[i]] = time.time()

        for i in range(len(client_ips)):
            self.threads[client_ips[i]].join()

    def no_offloading_train(self, client_ips):
        self.threads = {}
        for i in range(len(client_ips)):
            self.threads[client_ips[i]] = threading.Thread(target=self._thread_training_no_offloading,
                                                           args=(client_ips[i],), name=client_ips[i])
            fed_logger.info(str(client_ips[i]) + ' no offloading training start')
            self.threads[client_ips[i]].start()
        for i in range(len(client_ips)):
            self.threads[client_ips[i]].join()

    def no_edge_offloading_train(self, client_ips):
        self.threads = {}
        for i in range(len(client_ips)):
            self.threads[client_ips[i]] = threading.Thread(target=self._thread_training_offloading,
                                                           args=(client_ips[i],), name=client_ips[i])
            fed_logger.info(str(client_ips[i]) + ' no edge offloading training start')
            self.threads[client_ips[i]].start()
        for i in range(len(client_ips)):
            self.threads[client_ips[i]].join()

    def _thread_training_no_offloading(self, client_ip):
        pass

    def _thread_training_offloading(self, client_ip):
        # iteration = int((test_config.N / (test_config.K * test_config.B)))
        flag: bool = self.recv_msg(client_ip, config.mq_url, IterationFlagMessage.MESSAGE_TYPE).flag
        while flag:
            flag = self.recv_msg(client_ip, config.mq_url, IterationFlagMessage.MESSAGE_TYPE).flag
            if not flag:
                break
            # fed_logger.info(client_ip + " receiving local activations")
            msg: GlobalWeightMessage = self.recv_msg(client_ip, config.mq_url, GlobalWeightMessage.MESSAGE_TYPE)
            smashed_layers = msg.weights[0]
            labels = msg.weights[1]
            # fed_logger.info(client_ip + " training model")
            inputs, targets = smashed_layers.to(self.device), labels.to(self.device)
            if self.split_layers[config.CLIENTS_NAME_TO_INDEX[client_ip]] < len(
                    self.uninet.cfg) - 1:
                if self.optimizers.keys().__contains__(client_ip):
                    self.optimizers[client_ip].zero_grad()
            outputs = self.nets[client_ip](inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            if self.split_layers[config.CLIENTS_NAME_TO_INDEX[client_ip]] < len(
                    self.uninet.cfg) - 1:
                if self.optimizers.keys().__contains__(client_ip):
                    self.optimizers[client_ip].step()

            # Send gradients to edge
            fed_logger.info(client_ip + " sending gradients")
            msg = GlobalWeightMessage([inputs.grad])
            self.send_msg(client_ip, config.mq_url, msg)

        fed_logger.info(str(client_ip) + 'no edge offloading training end')
        return 'Finish'

    def _thread_edge_training(self, client_ip):
        # iteration = int((test_config.N / (test_config.K * test_config.B)))
        server_exchange = 'server.' + client_ip
        edge_exchange = 'edge.' + client_ip
        i = 0
        flag: bool = self.recv_msg(server_exchange, config.mq_url, IterationFlagMessage.MESSAGE_TYPE).flag
        i += 1
        fed_logger.info(Fore.RED + f"{flag}" + Fore.RESET)
        if not flag:
            fed_logger.info(str(client_ip) + ' offloading training end')
            return 'Finish'
        while flag:

            # fed_logger.info(client_ip + " receiving local activations")
            if self.split_layers[config.CLIENTS_NAME_TO_INDEX[client_ip]][1] < len(
                    self.uninet.cfg) - 1:
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
            i += 1

        fed_logger.info(str(client_ip) + ' offloading training end')
        return 'Finish'

    def aggregate(self, client_ips, aggregate_method, eweights):
        w_local_list = []

        # Log the incoming parameters for debugging
        fed_logger.info(
            f"Starting aggregation with client_ips: {client_ips}, aggregate_method: {aggregate_method}, eweights: {eweights}")

        for i in range(len(eweights)):

            if self.offload:
                sp = self.split_layers[i]
                if self.edge_based:
                    sp = self.split_layers[i][0]

                # Log the split layer position
                fed_logger.info(f"Split layer position (sp): {sp}, model_len: {config.model_len}")

                if sp != (config.model_len - 1):
                    # Log the model states being concatenated
                    fed_logger.info(f"Concatenating weights for client {client_ips[i]}")

                    w_local = (
                        model_utils.concat_weights(self.uninet.state_dict(), eweights[i],
                                                   self.nets[client_ips[i]].state_dict()),
                        config.N / config.K
                    )
                    w_local_list.append(w_local)
                else:
                    w_local = (eweights[i], config.N / config.K)
            else:
                w_local = (eweights[i], config.N / config.K)

            # Log the local weights being added
            fed_logger.info(f"Adding local weight for index {i}: {w_local}")
            w_local_list.append(w_local)

        # Initialize zero model and log before aggregation
        zero_model = model_utils.zero_init(self.uninet).state_dict()
        fed_logger.info(f"Zero model initialized for aggregation.")

        # Perform aggregation
        aggregated_model = aggregate_method(zero_model, w_local_list, config.N)
        fed_logger.info(f"Aggregation completed. Aggregated model keys: {aggregated_model.keys()}")

        self.uninet.load_state_dict(aggregated_model)
        return aggregated_model

    def test_network(self, connection_ips):
        """
        send message to test network speed
        """
        # Network test_app
        self.net_threads = {}
        for i in range(len(connection_ips)):
            self.net_threads[connection_ips[i]] = threading.Thread(target=self._thread_network_testing,
                                                                   args=(connection_ips[i],), name=connection_ips[i])
            self.net_threads[connection_ips[i]].start()

        for i in range(len(connection_ips)):
            self.net_threads[connection_ips[i]].join()

    def _thread_network_testing(self, connection_ip):
        network_time_start = time.time()
        msg = NetworkTestMessage([self.uninet.cpu().state_dict()])
        self.send_msg(connection_ip, config.mq_url, msg)
        fed_logger.info("server test network sent")
        server_exchange = 'server.' + connection_ip
        msg: NetworkTestMessage = self.recv_msg(server_exchange, config.mq_url, NetworkTestMessage.MESSAGE_TYPE)
        fed_logger.info("server test network received")
        network_time_end = time.time()
        self.edge_bandwidth[connection_ip] = data_utils.sizeofmessage(msg.weights) / (
                network_time_end - network_time_start)

    def client_network(self, edge_ips):
        """
        receive client network speed
        """

        for i in edge_ips:
            server_exchange = 'server.' + i
            msg = self.recv_msg(server_exchange, config.mq_url, JsonMessage.MESSAGE_TYPE).data[0]
            for k in msg.keys():
                self.client_bandwidth[k] = msg[k]

    def send_split_layers_config(self):
        """
        send splitting data
        """
        self.scatter(SplitLayerConfigMessage(self.split_layers))

    def send_split_layers_config_to_edges(self):
        """
        send splitting data
        """
        self.scatter(SplitLayerConfigMessage(self.split_layers))

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

    def e_energy_tt(self, client_ips):
        """
        Returns: average energy consumption of clients
        """
        energy_tt_list = []
        for edge in list(config.EDGE_SERVER_LIST):
            msg: GlobalWeightMessage = self.recv_msg(edge, config.mq_url,
                                                     GlobalWeightMessage.MESSAGE_TYPE)
            for i in range(len(config.EDGE_NAME_TO_CLIENTS_NAME[edge])):
                energy_tt_list.append(msg.weights[0][i])
        return energy_tt_list

    def c_local_weights(self, client_ips):
        cweights = []
        for i in range(len(client_ips)):
            msg: GlobalWeightMessage = self.recv_msg(config.SERVER_INDEX_TO_NAME[config.index], config.mq_url,
                                                     GlobalWeightMessage.MESSAGE_TYPE)
            self.tt_end[client_ips[i]] = time.time()
            cweights.append(msg.weights[0])
        return cweights

    def edge_offloading_global_weights(self):
        """
        send global weights
        """
        msg = GlobalWeightMessage([self.uninet.state_dict()])
        self.scatter(msg)

    def no_offloading_global_weights(self):
        msg = GlobalWeightMessage([self.uninet.state_dict()])
        self.scatter(msg)

    def cluster(self, options: dict):
        self.group_labels = fl_method_parser.fl_methods.get(options.get('clustering'))()

    def split(self, state, options: dict):
        self.split_layers = fl_method_parser.fl_methods.get(options.get('splitting'))(state, self.group_labels)
        fed_logger.info('Next Round OPs: ' + str(self.split_layers))

    def edge_based_state(self):
        state = []
        for i in self.client_bandwidth:
            state.append(self.client_bandwidth[i])
        for i in self.edge_bandwidth:
            state.append(self.edge_bandwidth[i])
        #
        # edge_offloading = []
        # server_offloading = 0
        # for i in range(len(config.EDGE_NAME_TO_CLIENTS_NAME)):
        #     edge_offloading.append(0)
        #     for j in range(len(config.EDGE_NAME_TO_CLIENTS_NAME.get((list(config.EDGE_NAME_TO_CLIENTS_NAME.keys()))[i]))):
        #         split_key = config.CLIENTS_NAME_TO_INDEX.get(config.EDGE_NAME_TO_CLIENTS_NAME.get(list(config.EDGE_NAME_TO_CLIENTS_NAME.keys())[i])[j])
        #         if self.split_layers[split_key][0] < self.split_layers[split_key][1]:
        #             edge_offloading[i] += 1
        #         if self.split_layers[split_key][1] < model_utils.get_unit_model_len() - 1:
        #             server_offloading += 1
        #     state.append(edge_offloading[i])
        # state.append(server_offloading)

        # for i in range(len(offloading)):
        #     state.append(offloading[i][0])
        #     state.append(offloading[i][1])
        return state

    def edge_based_reward_function_data(self, energy_tt_list, total_tt):
        energy = 0
        data = []
        tt = []
        for et in energy_tt_list:
            energy += et[0]
            tt.append(et[1])
        energy /= len(energy_tt_list)
        data.append(energy)
        data.append(total_tt)
        data.extend(tt)
        return data

    def call_aggregation(self, options: dict, eweights):
        method = fl_method_parser.fl_methods.get(options.get('aggregation'))
        if method is None:
            fed_logger.error("aggregate method is none")
        self.aggregate(config.CLIENTS_LIST, method, eweights)

    def dec_aggregate(self, client_local_weights: dict[str, BaseModel]) -> None:
        zero_model = model_utils.zero_init(self.uninet).state_dict()
        w_local_list = self._concat_neighbor_local_weights(client_local_weights)
        aggregated_model = self.aggregator.aggregate(zero_model, w_local_list)
        self.uninet.load_state_dict(aggregated_model)

    def _concat_neighbor_local_weights(self, client_local_weights) -> list:
        w_local_list = []
        client_neighbors = self.get_neighbors([NodeType.CLIENT])
        for neighbor in client_neighbors:
            if HTTPCommunicator.get_is_leader(neighbor):
                HTTPCommunicator.set_leader(neighbor, neighbor.ip, neighbor.port, False)
                fed_logger.info(f"client_local_weights: {client_local_weights.keys()}")
                w_local = (client_local_weights[str(neighbor)], config.N / len(client_neighbors))
                w_local_list.append(w_local)
        return w_local_list

    def scatter(self, msg: BaseMessage):
        list1 = config.CLIENTS_LIST
        if self.edge_based:
            list1 = config.EDGE_SERVER_LIST
            for i in list1:
                self.send_msg(i, config.mq_url, msg)

        else:
            for i in list1:
                self.send_msg(i, config.mq_url, msg)

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

    def receive_random_client_weights(self):
        client_local_weights = {}
        for neighbor in self.get_neighbors([NodeType.CLIENT]):
            is_leader = HTTPCommunicator.get_is_leader(neighbor)
            if is_leader:
                msg: GlobalWeightMessage = self.recv_msg(neighbor.get_exchange_name(), config.current_node_mq_url,
                                                         GlobalWeightMessage.MESSAGE_TYPE)
                client_local_weights[str(neighbor)] = msg.weights[0]
        return client_local_weights

    def choose_random_client_per_cluster(self):
        clusters = defaultdict(list)
        neighbors = self.get_neighbors([NodeType.CLIENT])

        # Step 2: Group clients by cluster
        for neighbor in neighbors:
            try:
                fed_logger.info(f"Fetching cluster info for neighbor: {neighbor}")
                cluster_info = HTTPCommunicator.get_cluster(neighbor)

                if not cluster_info:
                    fed_logger.warning(f"No cluster info returned for neighbor: {neighbor}")
                    continue

                cluster_id = cluster_info.get('cluster')
                fed_logger.info(f"Cluster info for neighbor {neighbor}: {cluster_info}")

                if cluster_id:
                    clusters[cluster_id].append(neighbor)
                    fed_logger.info(f"Added neighbor {neighbor} to cluster {cluster_id}")
                else:
                    fed_logger.warning(f"No valid cluster_id found for neighbor: {neighbor}")

            except Exception as e:
                fed_logger.error(f"Failed to get cluster info for neighbor {neighbor}: {e}")

        fed_logger.info(f"Cluster groups formed: {clusters}")

        # Step 3: Select random client per cluster and set leader
        random_clients = {}
        for cluster, clients in clusters.items():
            if clients:
                try:
                    fed_logger.info(f"Selecting random client for cluster {cluster} from clients: {clients}")
                    random_client = random.choice(clients)
                    random_clients[cluster] = random_client
                    fed_logger.info(f"Setting leader for cluster {cluster}: {random_client}")
                    HTTPCommunicator.set_leader(random_client, random_client.ip, random_client.port, True)
                    fed_logger.info(f"Successfully set leader for cluster {cluster}: {random_client}")

                except Exception as e:
                    fed_logger.error(f"Failed to set leader for cluster {cluster}: {e}")

        fed_logger.info(f"Final random clients per cluster: {random_clients}")

        return random_clients