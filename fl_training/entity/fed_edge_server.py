from torch import optim, nn

from config import config
from fl_training.interface.fed_edgeserver_interface import FedEdgeServerInterface
from util import model_utils, message_utils


class FedEdgeServer(FedEdgeServerInterface):
    def initialize(self, split_layers, offload, first, LR):
        if offload or first:
            self.split_layers = split_layers
            self.nets = {}
            self.optimizers = {}
            for i in range(len(split_layers)):
                client_ip = config.CLIENTS_LIST[i]
                if split_layers[i] < len(
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

        msg = [message_utils.initial_global_weights_server_to_client, self.uninet.state_dict()]
        for i in self.client_socks:
            self.send_msg(self.client_socks[i], msg)

    def train(self, thread_number, client_ips):
        pass

    def aggregate(self, client_ips, aggregate_method):
        pass
