import http
import threading

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import config
from app.entity.http_communicator import HTTPCommunicator
from app.entity.node_coordinate import NodeCoordinate
from app.entity.node_type import NodeType

from app.entity.node_identifier import NodeIdentifier


class Node:
    _app: FastAPI
    _neighbors: list[NodeIdentifier]
    _node_type: NodeType
    node_coordinate: NodeCoordinate

    def __init__(self, ip: str, port: int, node_type: NodeType):
        self._server_started = False
        self.ip = ip
        self.port = port
        self._node_type = node_type
        self._neighbors = []
        self.node_coordinate = None

        self._app = FastAPI()
        self._setup_routes()
        self._start_server_in_thread(port)

    def __str__(self):
        return f'{self.ip}:{self.port}'

    def _setup_routes(self):
        self._app.add_route("/get-node-type", self.get_node_type, methods=["GET"])
        self._app.add_route("/get-rabbitmq-url", self.get_rabbitmq_url, methods=["GET"])
        self._app.add_route("/get-node-coordinate", self.get_node_coordinate, methods=["GET"])

    async def get_node_type(self, _: Request):
        return JSONResponse({'node_type': self._node_type.name}, http.HTTPStatus.OK)

    @staticmethod
    async def get_rabbitmq_url(_: Request):
        return JSONResponse({'rabbitmq_url': config.current_node_mq_url}, http.HTTPStatus.OK)

    async def get_node_coordinate(self, _: Request):
        if self.node_coordinate is None:
            return JSONResponse({'error': 'Node coordinate not set'}, http.HTTPStatus.NOT_FOUND)
        return JSONResponse({
            'latitude': self.node_coordinate.latitude,
            'longitude': self.node_coordinate.longitude,
            'altitude': self.node_coordinate.altitude,
            'seconds_since_start': self.node_coordinate.seconds_since_start
        }, http.HTTPStatus.OK)

    def update_coordinates(self, new_latitude, new_longitude, new_altitude, new_seconds_since_start):
        self.node_coordinate = NodeCoordinate(
            latitude=new_latitude,
            longitude=new_longitude,
            altitude=new_altitude,
            seconds_since_start=new_seconds_since_start
        )

    def add_neighbor(self, node_id: NodeIdentifier):
        if node_id not in self._neighbors:
            self._neighbors.append(node_id)

    def add_neighbors(self, node_ids: list[NodeIdentifier]):
        for node_id in node_ids:
            self.add_neighbor(node_id)

    def get_neighbors(self, node_types: list[NodeType] = None) -> list[NodeIdentifier]:
        if not node_types:
            return self._neighbors
        return [node for node in self._neighbors if HTTPCommunicator.get_node_type(node) in node_types]

    def get_exchange_name(self) -> str:
        return f"{self.ip}:{self.port}"

    def _run_server(self, port: int):
        uvicorn.run(self._app, host="0.0.0.0", port=port)

    def _start_server_in_thread(self, port: int):
        if self._server_started:
            return
        self._server_started = True
        server_thread = threading.Thread(target=self._run_server, args=(port,))
        server_thread.start()
