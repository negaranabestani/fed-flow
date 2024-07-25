from typing import List

from app.entity.aggregator import Aggregator
from app.entity.aggregator_config import AggregatorConfig

from app.entity.neighbor import Neighbor


class Node:
    def __init__(self, ip: str, port: int, aggregator_config: AggregatorConfig):
        self.ip = ip
        self.port = port
        self.neighbors: List[Neighbor] = []
        self.aggregator = Aggregator(aggregator_config)

    def add_neighbor(self, neighbor_ip: str, neighbor_port: int):
        neighbor = Neighbor(neighbor_ip, neighbor_port)
        if neighbor not in self.neighbors:
            self.neighbors.append(neighbor)

    def get_neighbors(self) -> List[Neighbor]:
        return self.neighbors

    # def broadcast(self, message: str):
    #     for neighbor in self.neighbors:
    #         # Send message to neighbor
