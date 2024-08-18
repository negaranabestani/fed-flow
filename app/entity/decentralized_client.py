import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from app.config.logger import fed_logger
from app.dto.message import GlobalWeightMessage, NetworkTestMessage, SplitLayerConfigMessage, IterationFlagMessage
from app.dto.received_message import ReceivedMessage
from app.entity.communicator import Communicator
from app.entity.fed_base_node_interface import FedBaseNodeInterface
from app.entity.node import Node
from app.entity.node_type import NodeType
from app.util import model_utils


# noinspection PyTypeChecker
class DecentralizedClient(FedBaseNodeInterface):

    def __init__(self, ip: str, port: int, model_name, dataset, train_loader, LR):
        Node.__init__(self, ip, port, NodeType.CLIENT)
        Communicator.__init__(self)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.edge_based = True
        self.dataset = dataset
        self.train_loader = train_loader
        self.split_layers = None
        self.uninet = model_utils.get_model('Unit', None, self.device, True)
        self.net = self.uninet
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=LR, momentum=0.9)
        self.edge_based = False

    def initialize(self, split_layer, LR):

        self.split_layers = split_layer

        fed_logger.debug('Building Model.')
        self.net = model_utils.get_model('Client', self.split_layers, self.device, self.edge_based)
        fed_logger.debug(self.net)
        self.criterion = nn.CrossEntropyLoss()
        if len(list(self.net.parameters())) != 0:
            self.optimizer = optim.SGD(self.net.parameters(), lr=LR,
                                       momentum=0.9)

    def gather_global_weights(self):
        msgs: list[ReceivedMessage] = self.gather_msgs(GlobalWeightMessage.MESSAGE_TYPE, [NodeType.EDGE])
        msg: GlobalWeightMessage = msgs[0].message
        pweights = model_utils.split_weights_client(msg.weights[0], self.net.state_dict())
        self.net.load_state_dict(pweights)

    def scatter_network_speed_to_edges(self):
        msg = NetworkTestMessage([self.uninet.cpu().state_dict()])
        self.scatter_msg(msg, [NodeType.EDGE])
        fed_logger.info("test network sent")

        _ = self.gather_msgs(NetworkTestMessage.MESSAGE_TYPE, [NodeType.EDGE])
        fed_logger.info("test network received")

    def gather_split_config(self):
        msgs = self.gather_msgs(SplitLayerConfigMessage.MESSAGE_TYPE, [NodeType.EDGE])
        msg: SplitLayerConfigMessage = msgs[0].message
        self.split_layers = msg.data

    def start_offloading_train(self):
        self.net.to(self.device)
        self.net.train()
        i = 0
        if self.split_layers == model_utils.get_unit_model_len() - 1:
            fed_logger.info("no offloding training start----------------------------")
            self.scatter_msg(IterationFlagMessage(False), [NodeType.EDGE])
            i += 1
            for batch_idx, (inputs, targets) in enumerate(tqdm(self.train_loader)):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

        if self.split_layers < model_utils.get_unit_model_len() - 1:
            fed_logger.info(f"offloding training start {self.split_layers}----------------------------")
            self.scatter_msg(IterationFlagMessage(True), [NodeType.EDGE])
            i += 1

            for batch_idx, (inputs, targets) in enumerate(tqdm(self.train_loader)):

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if self.optimizer is not None:
                    self.optimizer.zero_grad()
                outputs = self.net(inputs)

                self.scatter_msg(IterationFlagMessage(True), [NodeType.EDGE])

                msg = GlobalWeightMessage([outputs.cpu(),
                                           targets.cpu()])
                self.scatter_msg(msg, [NodeType.EDGE])

                fed_logger.info("receiving gradients")
                msgs: list[ReceivedMessage] = self.gather_msgs(GlobalWeightMessage.MESSAGE_TYPE, [NodeType.EDGE])
                msg: GlobalWeightMessage = msgs[0].message
                gradients = msg.weights[0].to(self.device)
                fed_logger.info("received gradients")

                outputs.backward(gradients)
                if self.optimizer is not None:
                    self.optimizer.step()
                i += 1
            self.scatter_msg(IterationFlagMessage(False), [NodeType.EDGE])

    def scatter_local_weights(self):
        self.scatter_msg(GlobalWeightMessage([self.net.cpu().state_dict()]), [NodeType.EDGE])
