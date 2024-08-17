import time

import requests

from app.config.logger import fed_logger
from app.entity.node_type import NodeType
from app.entity.node_identifier import NodeIdentifier


class HTTPCommunicator:
    MAX_RETRIES = 5
    WAIT_DURATION_SECONDS = 5

    @staticmethod
    def _wait_for_neighbor_to_get_ready(node_identifier: NodeIdentifier):
        for i in range(HTTPCommunicator.MAX_RETRIES):
            try:
                requests.get(f"http://{node_identifier.ip}:{node_identifier.port}/get-node-type")
                return
            except requests.exceptions.ConnectionError:
                pass
            fed_logger.info(
                f"Node {node_identifier} is not ready, waiting for {HTTPCommunicator.WAIT_DURATION_SECONDS} seconds")
            time.sleep(HTTPCommunicator.WAIT_DURATION_SECONDS)
        raise ConnectionError(f"Node {node_identifier} is not ready")

    @staticmethod
    def get_node_type(node_identifier: NodeIdentifier) -> NodeType:
        try:
            HTTPCommunicator._wait_for_neighbor_to_get_ready(node_identifier)
        except ConnectionError:
            return NodeType.UNKNOWN
        request_url = f"http://{node_identifier.ip}:{node_identifier.port}/get-node-type"
        response = requests.get(request_url)
        return NodeType.from_value(response.json()['node_type'])

    @staticmethod
    def get_rabbitmq_url(node_identifier: NodeIdentifier) -> str:
        HTTPCommunicator._wait_for_neighbor_to_get_ready(node_identifier)
        request_url = f"http://{node_identifier.ip}:{node_identifier.port}/get-rabbitmq-url"
        response = requests.get(request_url)
        return response.json()['rabbitmq_url']

    @staticmethod
    def get_node_coordinate(node_identifier: NodeIdentifier) -> dict:
        HTTPCommunicator._wait_for_neighbor_to_get_ready(node_identifier)
        request_url = f"http://{node_identifier.ip}:{node_identifier.port}/get-node-coordinate"
        response = requests.get(request_url)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(
                f"Failed to get coordinates for node {node_identifier}, status code: {response.status_code}")
