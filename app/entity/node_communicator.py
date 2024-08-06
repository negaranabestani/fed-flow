import time

import requests

from app.entity.node import NodeIdentifier, NodeType


class NodeCommunicator:
    MAX_RETRIES = 5
    WAIT_DURATION_SECONDS = 20

    @staticmethod
    def _wait_for_neighbor_to_get_ready(node_identifier: NodeIdentifier):
        for i in range(NodeCommunicator.MAX_RETRIES):
            try:
                requests.get(f"http://{node_identifier.ip}:{node_identifier.port}/get-node-type")
                return
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(NodeCommunicator.WAIT_DURATION_SECONDS)
        raise ConnectionError(f"Node {node_identifier} is not ready")

    @staticmethod
    def get_node_type(node_identifier: NodeIdentifier) -> NodeType:
        NodeCommunicator._wait_for_neighbor_to_get_ready(node_identifier)
        request_url = f"http://{node_identifier.ip}:{node_identifier.port}/get-node-type"
        response = requests.get(request_url)
        return NodeType.from_value(response.json()['node_type'])

    @staticmethod
    def get_rabbitmq_url(node_identifier: NodeIdentifier) -> str:
        NodeCommunicator._wait_for_neighbor_to_get_ready(node_identifier)
        request_url = f"http://{node_identifier.ip}:{node_identifier.port}/get-rabbitmq-url"
        response = requests.get(request_url)
        return response.json()['rabbitmq_url']
