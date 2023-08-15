import threading
import time

from torch import optim, nn

from config import config
from config.logger import fed_logger
from fl_training.interface.fed_edgeserver_interface import FedEdgeServerInterface
from util import message_utils, model_utils


class FedEdgeServer(FedEdgeServerInterface):
    def initialize(self, split_layers, LR, client_ips):
        self.split_layers = split_layers
        self.optimizers = {}
        for i in range(len(split_layers)):
            if client_ips.__contains__(config.CLIENTS_LIST[i]):
                client_ip = config.CLIENTS_LIST[i]
                if split_layers[i][0] < len(
                        self.uninet.cfg) - 1:  # Only offloading client need initialize optimizer in server
                    self.nets[client_ip] = model_utils.get_model('Edge', split_layers[i], self.device)

                    # offloading weight in server also need to be initialized from the same global weight
                    cweights = model_utils.get_model('Client', split_layers[i], self.device).state_dict()

                    pweights = model_utils.split_weights_edgeserver(self.uninet.state_dict(), cweights,
                                                                    self.nets[client_ip].state_dict())
                    self.nets[client_ip].load_state_dict(pweights)

                    self.optimizers[client_ip] = optim.SGD(self.nets[client_ip].parameters(), lr=LR,
                                                           momentum=0.9)
                else:
                    self.nets[client_ip] = model_utils.get_model('Edge', split_layers[i], self.device)
        self.criterion = nn.CrossEntropyLoss()

    def aggregate(self, client_ips, aggregate_method):
        pass

    def forward_propagation(self, client_ip):
        flag = self.recv_msg(self.socks[config.CLIENT_MAP[client_ip]],
                             message_utils.local_iteration_flag_client_to_edge)[1]
        flmsg = [message_utils.local_iteration_flag_edge_to_server, flag]
        self.central_server_communicator.send_msg(self.central_server_communicator.sock, flmsg)
        while flag:
            flag = self.recv_msg(self.socks[config.CLIENT_MAP[client_ip]],
                                 message_utils.local_iteration_flag_client_to_edge)[1]
            flmsg = [message_utils.local_iteration_flag_edge_to_server, flag]
            self.central_server_communicator.send_msg(self.central_server_communicator.sock, flmsg)
            if not flag:
                break
            # fed_logger.info(client_ip + " receiving local activations")
            msg = self.recv_msg(self.socks[client_ip],
                                message_utils.local_activations_client_to_edge)
            smashed_layers = msg[1]
            labels = msg[2]
            # fed_logger.info(client_ip + " training model forward")
            inputs, targets = smashed_layers.to(self.device), labels.to(self.device)
            self.optimizers[client_ip].zero_grad()
            outputs = self.nets[client_ip](inputs)


            # fed_logger.info(client_ip + " sending local activations")
            msg = [message_utils.local_activations_edge_to_server, outputs.cpu(), targets.cpu()]
            self.central_server_communicator.send_msg(self.central_server_communicator.sock, msg)
            # fed_logger.info(client_ip + " receiving gradients")
            msg = self.central_server_communicator.recv_msg(self.central_server_communicator.sock,
                                                            message_utils.server_gradients_server_to_edge + str(
                                                                self.ip))
            gradients = msg[1].to(self.device)
            # fed_logger.info(client_ip + " training model backward")
            outputs.backward(gradients)
            self.optimizers[client_ip].step()
            # fed_logger.info(client_ip + " sending gradients")
            msg = [message_utils.server_gradients_edge_to_client + str(config.CLIENT_MAP[client_ip]), inputs.grad]
            self.send_msg(self.socks[config.CLIENT_MAP[client_ip]], msg)

        fed_logger.info(str(client_ip) + ' offloading training end')

    def backward_propagation(self, outputs, client_ip, inputs):
        pass

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
        msg = self.central_server_communicator.recv_msg(self.central_server_communicator.sock,
                                                        message_utils.test_server_network)
        msg = [message_utils.test_server_network, self.uninet.cpu().state_dict()]
        self.central_server_communicator.send_msg(self.central_server_communicator.sock, msg)

    def client_network(self):
        """
        send client network speed to central server
        """
        msg = [message_utils.client_network, self.client_bandwidth]
        self.central_server_communicator.send_msg(self.central_server_communicator.sock, msg)

    def split_layer(self, client_ips):
        """
        receive send splitting data to clients
        """
        msg = self.central_server_communicator.recv_msg(self.central_server_communicator.sock,
                                                        message_utils.split_layers_server_to_edge)
        self.split_layers = msg[1]
        for i in range(len(self.split_layers)):
            if client_ips.__contains__(config.CLIENTS_LIST[i]):
                client_ip = config.CLIENTS_LIST[i]
                msg = [message_utils.split_layers_edge_to_client, self.split_layers[i]]
                self.send_msg(self.socks[client_ip], msg)

    def local_weights(self, client_ip):
        """
        receive and send final weights for aggregation
        """
        cweights = self.recv_msg(self.socks[client_ip],
                                 message_utils.local_weights_client_to_edge)[1]

        msg = [message_utils.local_weights_edge_to_server, cweights]
        self.central_server_communicator.send_msg(self.central_server_communicator.sock, msg)

    def global_weights(self, client_ips: []):
        """
        receive and send global weights
        """
        weights = \
            self.central_server_communicator.recv_msg(self.central_server_communicator.sock,
                                                      message_utils.initial_global_weights_server_to_edge)[1]
        for i in range(len(self.split_layers)):
            if client_ips.__contains__(config.CLIENTS_LIST[i]):
                cweights = model_utils.get_model('Client', self.split_layers[i], self.device).state_dict()

                print("-------------------" + str(self.nets))
                pweights = model_utils.split_weights_edgeserver(weights, cweights,
                                                                self.nets[config.CLIENTS_LIST[i]].state_dict())
                self.nets[config.CLIENTS_LIST[i]].load_state_dict(pweights)

        msg = [message_utils.initial_global_weights_edge_to_client, weights]
        self.scatter(msg)

    def thread_training(self, client_ip):
        self.forward_propagation(client_ip)
        self.local_weights(client_ip)
