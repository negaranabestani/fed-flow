import http
import threading
from dataclasses import dataclass
from enum import Enum

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


@dataclass
class NodeIdentifier:
    ip: str
    port: int


class NodeType(Enum):
    CLIENT = "client"
    EDGE = "edge"
    SERVER = "server"


class Node:
    _app: FastAPI
    _neighbors: list[NodeIdentifier]
    _node_type: NodeType

    def __init__(self, ip: str, port: int, node_type: NodeType):
        self.ip = ip
        self.port = port
        self._node_type = node_type
        self._neighbors = []
        self._app = FastAPI()
        self._setup_routes()
        self._start_server_in_thread(port)

    def _setup_routes(self):
        self._app.add_route("/get-node-type", self.get_node_type, methods=["GET"])

    async def get_node_type(self, _: Request):
        return JSONResponse({'node_type': self._node_type.name}, http.HTTPStatus.OK)

    def add_neighbor(self, node_id: NodeIdentifier):
        if node_id not in self._neighbors:
            self._neighbors.append(node_id)

    def add_neighbors(self, node_ids: list[NodeIdentifier]):
        for node_id in node_ids:
            self.add_neighbor(node_id)

    def get_neighbors(self) -> list[NodeIdentifier]:
        return self._neighbors

    def _run_server(self, port: int):
        uvicorn.run(self._app, host="0.0.0.0", port=port)

    def _start_server_in_thread(self, port: int):
        server_thread = threading.Thread(target=self._run_server, args=(port,))
        server_thread.start()
