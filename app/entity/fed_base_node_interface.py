from abc import ABC

from app.config import config
from app.dto.message import BaseMessage, GlobalWeightMessage, SplitLayerConfigMessage, MessageType
from app.dto.received_message import ReceivedMessage
from app.entity.communicator import Communicator
from app.entity.node import Node
from app.entity.node_type import NodeType
from app.entity.http_communicator import HTTPCommunicator


class FedBaseNodeInterface(ABC, Node, Communicator):
    def __init__(self, ip: str, port: int, node_type: NodeType):
        super().__init__(ip, port, node_type)
        self.uninet = None
        self.split_layers = None

    def scatter_msg(self, msg: BaseMessage, neighbors_types: list[NodeType] = None):
        for neighbor in self.get_neighbors():
            neighbor_type = HTTPCommunicator.get_node_type(neighbor)
            if neighbors_types is None or neighbor_type in neighbors_types:
                rabbitmq_url = HTTPCommunicator.get_rabbitmq_url(neighbor)
                self.send_msg(self.get_exchange_name(), rabbitmq_url, msg)

    def scatter_global_weights(self, neighbors_types: list[NodeType] = None):
        msg = GlobalWeightMessage([self.uninet.state_dict()])
        self.scatter_msg(msg, neighbors_types)

    def scatter_split_layers(self, neighbors_types: list[NodeType] = None):
        for neighbor in self.get_neighbors():
            neighbor_type = HTTPCommunicator.get_node_type(neighbor)
            if neighbors_types is None or neighbor_type in neighbors_types:
                msg = SplitLayerConfigMessage(self.split_layers[str(neighbor)])
                self.send_msg(self.get_exchange_name(), HTTPCommunicator.get_rabbitmq_url(neighbor), msg)

    def gather_msgs(self, msg_type: MessageType, neighbors_types: list[NodeType] = None) -> list[
            ReceivedMessage]:  # (ip, msg)
        messages = []

        for neighbor in self.get_neighbors():
            neighbor_type = HTTPCommunicator.get_node_type(neighbor)
            if neighbors_types is None or neighbor_type in neighbors_types:
                msg = self.recv_msg(neighbor.get_exchange_name(), config.current_node_mq_url, msg_type)
                messages.append(ReceivedMessage(msg, neighbor))
        return messages
