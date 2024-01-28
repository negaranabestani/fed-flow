import socket
import sys
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from app.entity.interface.fed_server_interface import FedServerInterface
from app.fl_method import fl_method_parser

sys.path.append('../../')
from app.util import message_utils, model_utils
from app.config import config
from app.config.logger import fed_logger

np.random.seed(0)
torch.manual_seed(0)


class FedServer(FedServerInterface):

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
        flag = self.recv_msg(self.edge_socks[socket.gethostbyname(client_ip)],
                             message_utils.local_iteration_flag_client_to_server + "_" + client_ip)[1]
        while flag:
            flag = self.recv_msg(self.edge_socks[socket.gethostbyname(client_ip)],
                                 message_utils.local_iteration_flag_client_to_server + "_" + client_ip)[1]
            if not flag:
                break
            # fed_logger.info(client_ip + " receiving local activations")
            msg = self.recv_msg(self.edge_socks[socket.gethostbyname(client_ip)],
                                message_utils.local_activations_client_to_server + "_" + client_ip)
            smashed_layers = msg[1]
            labels = msg[2]
            # fed_logger.info(client_ip + " training model")
            inputs, targets = smashed_layers.to(self.device), labels.to(self.device)
            if self.split_layers[config.CLIENTS_CONFIG[client_ip]] < len(
                    self.uninet.cfg) - 1:
                if self.optimizers.keys().__contains__(client_ip):
                    self.optimizers[client_ip].zero_grad()
            outputs = self.nets[client_ip](inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            if self.split_layers[config.CLIENTS_CONFIG[client_ip]] < len(
                    self.uninet.cfg) - 1:
                if self.optimizers.keys().__contains__(client_ip):
                    self.optimizers[client_ip].step()

            # Send gradients to edge
            # fed_logger.info(client_ip + " sending gradients")
            msg = [message_utils.server_gradients_server_to_client + str(client_ip), inputs.grad]
            self.send_msg(self.edge_socks[socket.gethostbyname(client_ip)], msg)

        fed_logger.info(str(client_ip) + 'no edge offloading training end')
        return 'Finish'

    def _thread_edge_training(self, client_ip):
        # iteration = int((test_config.N / (test_config.K * test_config.B)))
        flag = self.recv_msg(self.socks[socket.gethostbyname(client_ip)],
                             message_utils.local_iteration_flag_edge_to_server + "_" + client_ip)[1]
        while flag:
            flag = self.recv_msg(self.socks[socket.gethostbyname(client_ip)],
                                 message_utils.local_iteration_flag_edge_to_server + "_" + client_ip)[1]
            if not flag:
                break
            # fed_logger.info(client_ip + " receiving local activations")
            msg = self.recv_msg(self.socks[socket.gethostbyname(client_ip)],
                                message_utils.local_activations_edge_to_server + "_" + client_ip)
            smashed_layers = msg[1]
            labels = msg[2]
            # fed_logger.info(client_ip + " training model")
            inputs, targets = smashed_layers.to(self.device), labels.to(self.device)
            if self.split_layers[config.CLIENTS_CONFIG[client_ip]][1] < len(
                    self.uninet.cfg) - 1:
                if self.optimizers.keys().__contains__(client_ip):
                    self.optimizers[client_ip].zero_grad()
            outputs = self.nets[client_ip](inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            if self.split_layers[config.CLIENTS_CONFIG[client_ip]][1] < len(
                    self.uninet.cfg) - 1:
                if self.optimizers.keys().__contains__(client_ip):
                    self.optimizers[client_ip].step()

            # Send gradients to edge
            # fed_logger.info(client_ip + " sending gradients")
            msg = [message_utils.server_gradients_server_to_edge + str(client_ip), inputs.grad]
            self.send_msg(self.socks[socket.gethostbyname(client_ip)], msg)

        fed_logger.info(str(client_ip) + ' offloading training end')
        return 'Finish'

    def aggregate(self, client_ips, aggregate_method, eweights):
        w_local_list = []
        # fed_logger.info("aggregation start")
        for i in range(len(eweights)):
            # if self.split_layers[i] != (test_config.model_len - 1):
            #     w_local = (
            #         model_utils.concat_weights(self.uninet.state_dict(), eweights[i],
            #                                    self.nets[client_ips[i]].state_dict()),
            #         test_config.N / test_config.K)
            #     w_local_list.append(w_local)
            # else:
            w_local = (eweights[i], config.N / config.K)
            # print("------------------------"+str(eweights[i]))
            w_local_list.append(w_local)
        zero_model = model_utils.zero_init(self.uninet).state_dict()
        # fed_logger.info("calling aggregation method")
        aggregated_model = aggregate_method(zero_model, w_local_list, config.N)
        # fed_logger.info("aggregation method end")
        self.uninet.load_state_dict(aggregated_model)
        # fed_logger.info("aggregation end")
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
        msg = [message_utils.test_network, self.uninet.cpu().state_dict()]
        self.send_msg(self.edge_socks[socket.gethostbyname(connection_ip)], msg)
        msg = self.recv_msg(self.edge_socks[socket.gethostbyname(connection_ip)], message_utils.test_network)
        network_time_end = time.time()
        self.edge_bandwidth[connection_ip] = network_time_end - network_time_start

    def client_network(self, edge_ips):
        """
        receive client network speed
        """

        for i in edge_ips:
            msg = self.recv_msg(self.edge_socks[socket.gethostbyname(str(i))], message_utils.client_network)
            for k in msg[1].keys():
                self.client_bandwidth[k] = msg[1][k]

    def split_layer(self):
        """
        send splitting data
        """
        msg = [message_utils.split_layers, self.split_layers]
        self.scatter(msg)

    def e_local_weights(self, client_ips):
        """
        send final weights for aggregation
        """
        eweights = []
        for i in range(len(client_ips)):
            msg = self.recv_msg(self.socks[socket.gethostbyname(client_ips[i])],
                                message_utils.local_weights_edge_to_server + "_" + client_ips[i])
            self.tt_end[client_ips[i]] = time.time()
            eweights.append(msg[1])
        return eweights

    def e_energy_tt(self, client_ips):
        """
        Returns: average energy consumption of clients
        """
        energy_tt_list = []
        for edge in list(self.edge_socks.keys()):
            # fed_logger.info(f"receiving {socket.gethostbyaddr(edge)[0]}")
            msg = self.recv_msg(self.edge_socks[edge],
                                message_utils.energy_tt_edge_to_server)
            energy_tt_list.append(msg[1][0])
            energy_tt_list.append(msg[1][1])
        # fed_logger.info("ettlist:" + str(energy_tt_list))
        return energy_tt_list

    def c_local_weights(self, client_ips):
        cweights = []
        # msg = self.recv_msg(self.edge_socks[socket.gethostbyname(client_ips[0])],
        #                     message_utils.local_iteration_flag_client_to_server)
        # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"+str(msg[1]))
        for i in range(len(client_ips)):
            msg = self.recv_msg(self.edge_socks[socket.gethostbyname(client_ips[i])],
                                message_utils.local_weights_client_to_server)
            # fed_logger.info(f"cw received {client_ips[i]}")
            self.tt_end[client_ips[i]] = time.time()
            cweights.append(msg[1])
        return cweights

    def edge_offloading_global_weights(self):
        """
        send global weights
        """
        msg = [message_utils.initial_global_weights_server_to_edge, self.uninet.state_dict()]
        self.scatter(msg)

    def no_offloading_global_weights(self):
        msg = [message_utils.initial_global_weights_server_to_client, self.uninet.state_dict()]
        self.scatter(msg)

    def cluster(self, options: dict):
        self.group_labels = fl_method_parser.fl_methods.get(options.get('clustering'))()

    def split(self, state, options: dict):
        self.split_layers = fl_method_parser.fl_methods.get(options.get('splitting'))(state, self.group_labels)
        fed_logger.info('Next Round OPs: ' + str(self.split_layers))

    def edge_based_state(self, offloading, energy_tt_list):
        state = []
        energy = 0
        tt = []
        for et in energy_tt_list:
            energy += et[0]
            tt.append(et[1])
        energy /= len(config.CLIENTS_LIST)
        state.append(energy)
        state.append(max(tt))
        state.append(tt)
        # for i in range(config.S):
        #     state.append("utilization" + str(i))
        edge_offloading = []
        server_offloading = 0
        for i in range(len(config.EDGE_MAP)):
            edge_offloading.append(0)
            for j in range(len(config.EDGE_MAP.get((list(config.EDGE_MAP.keys()))[i]))):
                split_key = config.CLIENTS_CONFIG.get(config.EDGE_MAP.get(list(config.EDGE_MAP.keys())[i])[j])
                if self.split_layers[split_key][0] < model_utils.get_unit_model_len() - 1:
                    edge_offloading[i] += 1
                if self.split_layers[split_key][1] < model_utils.get_unit_model_len() - 1:
                    server_offloading += 1
            state.append(edge_offloading[i])
        state.append(server_offloading)

        for i in range(len(offloading)):
            state.append(offloading[i][0])
            state.append(offloading[i][1])
        return state
