from torch import optim, nn

from config import config
from fl_training.interface.fed_edgeserver_interface import FedEdgeServerInterface
from util import message_utils


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
