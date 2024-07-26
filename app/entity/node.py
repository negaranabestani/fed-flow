import threading
from typing import Dict
import uvicorn
from typing import List

from fastapi import FastAPI, Request, HTTPException

from app.entity.neighbor import Neighbor


class Node:
    _nodes: Dict[str, str] = {}

    def __init__(self, ip: str, port: int, device_type: str):
        self.ip = ip
        self.port = port
        self.device_type = device_type
        Node._nodes[f"{ip}:{port}"] = device_type

    @classmethod
    def get_device_type(cls, ip: str, port: int) -> str:
        key = f"{ip}:{port}"
        if key in cls._nodes:
            return cls._nodes[key]
        else:
            raise HTTPException(status_code=404, detail="Node not found")


# Create example nodes
Node('client1', 8080, 'Edge Server')

app = FastAPI()


def add_neighbor(self, neighbor_ip: str, neighbor_port: int):
    neighbor = Neighbor(neighbor_ip, neighbor_port)
    if neighbor not in self.neighbors:
        self.neighbors.append(neighbor)


def get_neighbors(self) -> List[Neighbor]:
    return self.neighbors


@app.get("/device_type")
async def get_device_type(request: Request):
    client_ip = request.client.host
    client_port = request.url.port

    if client_ip and client_port:
        device_type = Node.get_device_type(client_ip, client_port)
        return {"device_type": device_type}
    else:
        raise HTTPException(status_code=400, detail="Client IP or port not found in request")


def run_server(ip: str, port: int):
    uvicorn.run(app, host=ip, port=port)


def start_server_thread(ip: str, port: int):
    server_thread = threading.Thread(target=run_server, args=(ip, port))
    server_thread.start()
