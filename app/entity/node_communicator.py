import requests

from app.entity.node import NodeIdentifier, NodeType


class NodeCommunicator:
    @staticmethod
    def get_node_type(node_identifier: NodeIdentifier) -> NodeType:
        request_url = f"http://{node_identifier.ip}:{node_identifier.port}/get-node-type"
        response = requests.get(request_url)
        return NodeType.from_value(response.json()['node_type'])
