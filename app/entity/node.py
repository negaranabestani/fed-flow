from typing import List

from app.entity.neighbour import Neighbor


class Node:
    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port
        self.neighbors: List[Neighbor] = []

    def add_neighbor(self, neighbor_ip: str, neighbor_port: int):
        neighbor = Neighbor(neighbor_ip, neighbor_port)
        if neighbor not in self.neighbors:
            self.neighbors.append(neighbor)

    def get_neighbors(self) -> List[Neighbor]:
        return self.neighbors

    def broadcast(self, message: str):
        for neighbor in self.neighbors:
            self.send_message(message, neighbor.ip, neighbor.port)

    @staticmethod
    def send_message(message: str, receiver_ip: str, receiver_port: int):
        print(f"Sending message to {receiver_ip}:{receiver_port} - {message}")

    @staticmethod
    def receive_message(message: str, sender_ip: str, sender_port: int):
        print(f"Received message from {sender_ip}:{sender_port} - {message}")
