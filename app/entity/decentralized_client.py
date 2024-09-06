import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from app.config import config
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

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model_name = model_name
        self.edge_based = True
        self.dataset = dataset
        self.train_loader = train_loader
        self.split_layers = None
        self.uninet = model_utils.get_model('Unit', None, self.device, True)
        self.net = self.uninet
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, config.lr_step_size, config.lr_gamma)
        self.edge_based = False

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
            self.scheduler.step()

        elif self.split_layers < model_utils.get_unit_model_len() - 1:
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
            self.scheduler.step()
            self.scatter_msg(IterationFlagMessage(False), [NodeType.EDGE])

    def scatter_local_weights(self):
        self.scatter_msg(GlobalWeightMessage([self.net.cpu().state_dict()]), [NodeType.EDGE])
