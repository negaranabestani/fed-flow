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

    # def broadcast(self, message: str):
    #     for neighbor in self.neighbors:
