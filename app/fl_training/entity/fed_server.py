import sys
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from app.fl_method import fl_method_parser
from app.fl_training.interface.fed_server_interface import FedServerInterface

sys.path.append('../../../')
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
            if split_layers[i][0] < len(
                    self.uninet.cfg) - 1:  # Only offloading client need initialize optimizer in server
                self.nets[client_ip] = model_utils.get_model('Server', split_layers[i], self.device)

                # offloading weight in server also need to be initialized from the same global weight
                eweights = model_utils.get_model('Edge', split_layers[i], self.device).state_dict()
                cweights = model_utils.get_model('Client', split_layers[i], self.device).state_dict()

                pweights = model_utils.split_weights_server(self.uninet.state_dict(), cweights,
                                                            self.nets[client_ip].state_dict(), eweights)
                self.nets[client_ip].load_state_dict(pweights)

                self.optimizers[client_ip] = optim.SGD(self.nets[client_ip].parameters(), lr=LR,
                                                       momentum=0.9)
            else:
                self.nets[client_ip] = model_utils.get_model('Server', split_layers[i], self.device)
        self.criterion = nn.CrossEntropyLoss()

    def offloading_train(self, client_ips):
        self.threads = {}
        for i in range(len(client_ips)):
            fed_logger.info(str(client_ips[i]))
            self.threads[client_ips[i]] = threading.Thread(target=self._thread_training_offloading,
                                                           args=(client_ips[i],))
            fed_logger.info(str(client_ips[i]) + ' offloading training start')
            self.threads[client_ips[i]].start()
            self.tt_start[client_ips[i]] = time.time()

        for i in range(len(client_ips)):
            self.threads[client_ips[i]].join()

    def no_offloading_train(self, client_ips):
        self.threads = {}
        for i in range(len(client_ips)):
            self.threads[client_ips[i]] = threading.Thread(target=self._thread_training_no_offloading,
                                                           args=(client_ips[i],))
            fed_logger.info(str(client_ips[i]) + ' no offloading training start')
            self.threads[client_ips[i]].start()
        for i in range(len(client_ips)):
            self.threads[client_ips[i]].join()

    def _thread_training_no_offloading(self, client_ip):
        pass

    def _thread_training_offloading(self, client_ip):
        # iteration = int((test_config.N / (test_config.K * test_config.B)))
        flag = self.recv_msg(self.socks[config.CLIENT_MAP[client_ip]],
                             message_utils.local_iteration_flag_edge_to_server)[1]
        while flag:
            flag = self.recv_msg(self.socks[config.CLIENT_MAP[client_ip]],
                                 message_utils.local_iteration_flag_edge_to_server)[1]
            if not flag:
                break
            # fed_logger.info(client_ip + " receiving local activations")
            msg = self.recv_msg(self.socks[config.CLIENT_MAP[client_ip]],
                                message_utils.local_activations_edge_to_server)
            smashed_layers = msg[1]
            labels = msg[2]
            # fed_logger.info(client_ip + " training model")
            inputs, targets = smashed_layers.to(self.device), labels.to(self.device)
            self.optimizers[client_ip].zero_grad()
            outputs = self.nets[client_ip](inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizers[client_ip].step()

            # Send gradients to edge
            # fed_logger.info(client_ip + " sending gradients")
            msg = [message_utils.server_gradients_server_to_edge + str(config.CLIENT_MAP[client_ip]), inputs.grad]
            self.send_msg(self.socks[config.CLIENT_MAP[client_ip]], msg)


        fed_logger.info(str(client_ip) + ' offloading training end')
        return 'Finish'

    def aggregate(self, client_ips, aggregate_method, eweights):
        w_local_list = []
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
        aggregated_model = aggregate_method(zero_model, w_local_list, config.N)
        self.uninet.load_state_dict(aggregated_model)
        return aggregated_model

    def test_network(self, edge_ips):
        """
        send message to test_app network speed
        """
        # Network test_app
        self.net_threads = {}
        for i in range(len(edge_ips)):
            self.net_threads[edge_ips[i]] = threading.Thread(target=self._thread_network_testing,
                                                             args=(edge_ips[i],))
            self.net_threads[edge_ips[i]].start()

        for i in range(len(edge_ips)):
            self.net_threads[edge_ips[i]].join()

    def _thread_network_testing(self, edge_ip):
        network_time_start = time.time()
        msg = [message_utils.test_server_network, self.uninet.cpu().state_dict()]
        self.send_msg(self.socks[edge_ip], msg)
        msg = self.recv_msg(self.socks[edge_ip], message_utils.test_server_network)
        network_time_end = time.time()
        self.edge_bandwidth[edge_ip] = network_time_end - network_time_start

    def client_network(self, edge_ips):
        """
        receive client network speed
        """

        for i in edge_ips:
            msg = self.recv_msg(self.socks[str(i)], message_utils.client_network)
            for k in msg[1].keys():
                self.client_bandwidth[k] = msg[1][k]

    def split_layer(self):
        """
        send splitting data
        """
        msg = [message_utils.split_layers_server_to_edge, self.split_layers]
        self.scatter(msg)

    def e_local_weights(self, client_ips):
        """
        send final weights for aggregation
        """
        eweights = []
        for i in range(len(client_ips)):
            msg = self.recv_msg(self.socks[config.CLIENT_MAP[client_ips[i]]],
                                message_utils.local_weights_edge_to_server)
            self.tt_end[client_ips[i]] = time.time()
            eweights.append(msg[1])
        return eweights

    def c_local_weights(self, client_ips):
        cweights = []
        for i in range(len(client_ips)):
            msg = self.recv_msg(self.socks[client_ips[i]],
                                message_utils.local_weights_client_to_server)
            self.tt_end[client_ips[i]] = time.time()
            cweights.append(msg[1])
        return cweights

    def offloading_global_weights(self):
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

    def split(self, options: dict):
        self.split_layers = fl_method_parser.fl_methods.get(options.get('splitting'))(self.state, self.group_labels)
        fed_logger.info('Next Round OPs: ' + str(self.split_layer))
