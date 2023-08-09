import threading
import time

from torch import optim, nn

from config import config
from fl_training.interface.fed_edgeserver_interface import FedEdgeServerInterface
from util import message_utils, model_utils


class FedEdgeServer(FedEdgeServerInterface):
    def initialize(self, split_layers, LR, client_ips):
        self.split_layers = split_layers
        self.optimizers = {}
        for i in range(len(split_layers)):
            if vars(client_ips).__contains__(config.CLIENTS_LIST[i]):
                client_ip = config.CLIENTS_LIST[i]
                self.optimizers[client_ip] = optim.SGD(self.nets[client_ip].parameters(), lr=LR,
                                                       momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()

    def aggregate(self, client_ips, aggregate_method):
        pass

    def forward_propagation(self, client_ip):
        msg = self.recv_msg(self.socks[client_ip],
                            message_utils.local_activations_client_to_edge)
        smashed_layers = msg[1]
        labels = msg[2]
        inputs, targets = smashed_layers.to(self.device), labels.to(self.device)
        self.optimizers[client_ip].zero_grad()
        outputs = self.nets[client_ip](inputs)
        return outputs

    def backward_propagation(self, outputs, client_ip):
        msg = self.recv_msg(self.socks[self.ip],
                            message_utils.server_gradients_server_to_edge + str(self.ip))
        gradients = msg[1].to(self.device)
        outputs.backward(gradients)
        self.optimizers[client_ip].step()

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

    def thread_training(self, client_ip):
        outputs = self.forward_propagation(client_ip)
        self.backward_propagation(outputs, client_ip)
