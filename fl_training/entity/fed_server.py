import sys
import threading

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from fl_training.interface.fed_server_interface import FedServerInterface

sys.path.append('../../')
from util import model_utils, message_utils
from config import config
from config.logger import fed_logger

np.random.seed(0)
torch.manual_seed(0)


class FedServer(FedServerInterface):

    def initialize(self, split_layers, offload, first, LR):
        if offload or first:
            self.split_layers = split_layers
            self.nets = {}
            self.optimizers = {}
            for i in range(len(split_layers)):
                client_ip = config.CLIENTS_LIST[i]
                if split_layers[i] < len(
                        self.uninet.cfg) - 1:  # Only offloading client need initialize optimizer in server
                    self.nets[client_ip] = model_utils.get_model('Server', split_layers[i], self.device)

                    # offloading weight in server also need to be initialized from the same global weight
                    cweights = model_utils.get_model('Client', split_layers[i], self.device).state_dict()
                    pweights = model_utils.split_weights_server(self.uninet.state_dict(), cweights,
                                                                self.nets[client_ip].state_dict())
                    self.nets[client_ip].load_state_dict(pweights)

                    self.optimizers[client_ip] = optim.SGD(self.nets[client_ip].parameters(), lr=LR,
                                                           momentum=0.9)
                else:
                    self.nets[client_ip] = model_utils.get_model('Server', split_layers[i], self.device)
            self.criterion = nn.CrossEntropyLoss()

        msg = [message_utils.initial_global_weights_server_to_client, self.uninet.state_dict()]
        for i in self.client_socks:
            self.send_msg(self.client_socks[i], msg)

    def train(self, thread_number, client_ips):
        # Network test
        self.net_threads = {}
        for i in range(len(client_ips)):
            self.net_threads[client_ips[i]] = threading.Thread(target=self._thread_network_testing,
                                                               args=(client_ips[i],))
            self.net_threads[client_ips[i]].start()

        for i in range(len(client_ips)):
            self.net_threads[client_ips[i]].join()

        self.bandwidth = {}
        for s in self.client_socks:
            msg = self.recv_msg(self.client_socks[s], message_utils.test_network)
            self.bandwidth[msg[1]] = msg[2]

        # Training start
        self.threads = {}
        for i in range(len(client_ips)):
            if config.split_layer[i] == (config.model_len - 1):
                self.threads[client_ips[i]] = threading.Thread(target=self._thread_training_no_offloading,
                                                               args=(client_ips[i],))
                fed_logger.info(str(client_ips[i]) + ' no offloading training start')
                self.threads[client_ips[i]].start()
            else:
                fed_logger.info(str(client_ips[i]))
                self.threads[client_ips[i]] = threading.Thread(target=self._thread_training_offloading,
                                                               args=(client_ips[i],))
                fed_logger.info(str(client_ips[i]) + ' offloading training start')
                self.threads[client_ips[i]].start()

        for i in range(len(client_ips)):
            self.threads[client_ips[i]].join()

        self.ttpi = {}  # Training time per iteration
        for s in self.client_socks:
            msg = self.recv_msg(self.client_socks[s], message_utils.training_time_per_iteration_client_to_server)
            self.ttpi[msg[1]] = msg[2]

    def _thread_network_testing(self, client_ip):
        msg = self.recv_msg(self.client_socks[client_ip], message_utils.test_network)
        msg = [message_utils.test_network, self.uninet.cpu().state_dict()]
        self.send_msg(self.client_socks[client_ip], msg)

    def _thread_training_no_offloading(self, client_ip):
        pass

    def _thread_training_offloading(self, client_ip):
        iteration = int((config.N / (config.K * config.B)))
        for i in range(iteration):
            msg = self.recv_msg(self.client_socks[client_ip], message_utils.local_activations_client_to_server)
            smashed_layers = msg[1]
            labels = msg[2]

            inputs, targets = smashed_layers.to(self.device), labels.to(self.device)
            self.optimizers[client_ip].zero_grad()
            outputs = self.nets[client_ip](inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizers[client_ip].step()

            # Send gradients to client
            msg = [message_utils.server_gradients_server_to_client + str(client_ip), inputs.grad]
            self.send_msg(self.client_socks[client_ip], msg)

        fed_logger.info(str(client_ip) + ' offloading training end')
        return 'Finish'

    def aggregate(self, client_ips, aggregate_method):
        w_local_list = []
        for i in range(len(client_ips)):
            msg = self.recv_msg(self.client_socks[client_ips[i]], message_utils.local_weights_client_to_server)
            if config.split_layer[i] != (config.model_len - 1):
                w_local = (
                    model_utils.concat_weights(self.uninet.state_dict(), msg[1], self.nets[client_ips[i]].state_dict()),
                    config.N / config.K)
                w_local_list.append(w_local)
            else:
                w_local = (msg[1], config.N / config.K)
                w_local_list.append(w_local)
        zero_model = model_utils.zero_init(self.uninet).state_dict()
        aggregated_model = aggregate_method(zero_model, w_local_list, config.N)

        self.uninet.load_state_dict(aggregated_model)
        return aggregated_model
