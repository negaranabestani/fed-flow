from typing import List, Dict, Union
from enum import Enum


class NodeType(Enum):
    CLIENT = 1
    EDGE_SERVER = 2
    SERVER = 3


class Node:
    def __init__(self, node_id: int, ip: str, port: int, node_type: NodeType):
        self.node_id = node_id
        self.ip = ip
        self.port = port
        self.neighbors: List[Dict[str, Union[str, int, NodeType]]] = []
        self.kind = node_type

    def add_neighbor(self, neighbor_ip: str, neighbor_port: int, neighbor_type: NodeType):
        neighbor = {"ip": neighbor_ip, "port": neighbor_port, "type": neighbor_type}
        if neighbor not in self.neighbors:
            self.neighbors.append(neighbor)

    def get_neighbors(self) -> List[Dict[str, Union[str, int, NodeType]]]:
        return self.neighbors

    def broadcast(self, message: str):
        for neighbor in self.neighbors:
            self.send_message(message, neighbor["ip"], neighbor["port"])

    @staticmethod
    def send_message(message: str, receiver_ip: str, receiver_port: int):
        print(f"Sending message to {receiver_ip}:{receiver_port} - {message}")

    @staticmethod
    def receive_message(message: str, sender_ip: str, sender_port: int):
        print(f"Received message from {sender_ip}:{sender_port} - {message}")
