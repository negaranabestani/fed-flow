import io
import pickle
from abc import ABC
from typing import NewType, Final

import torch

MessageType = NewType("MessageType", str)


class BaseMessage(ABC):
    @staticmethod
    def create_message_by_type(msg_type: MessageType, data: any = None) -> 'BaseMessage':
        if msg_type == GlobalWeightMessage.MESSAGE_TYPE:
            return GlobalWeightMessage(data)
        elif msg_type == JsonMessage.MESSAGE_TYPE:
            return JsonMessage(data)
        else:
            raise ValueError(f"Unknown message type: {msg_type}")

    @staticmethod
    def deserialize(data: bytes, msg_type: MessageType) -> 'BaseMessage':
        cls = globals()[msg_type]
        deserialize = getattr(cls, 'deserialize')
        return deserialize(data, msg_type)

    def serialize(self) -> bytes:
        raise NotImplementedError

    def get_message_type(self) -> MessageType:
        raise NotImplementedError


class GlobalWeightMessage(BaseMessage):
    weights: list[dict]
    MESSAGE_TYPE: Final = MessageType('GlobalWeightMessage')

    def __init__(self, weights: list[dict]):
        self.weights = weights

    def serialize(self) -> bytes:
        weights = self.weights
        ll = []
        for o in weights:
            to_send = io.BytesIO()
            torch.save(o, to_send, _use_new_zipfile_serialization=False)
            to_send.seek(0)
            ll.append(bytes(to_send.read()))
        return pickle.dumps(ll)

    @staticmethod
    def deserialize(data: bytes, msg_type: MessageType) -> 'GlobalWeightMessage':
        fl: list[dict] = []
        ll = pickle.loads(data)
        for o in ll:
            fl.append(torch.load(io.BytesIO(o)))
        return GlobalWeightMessage(fl)

    def get_message_type(self) -> MessageType:
        return self.MESSAGE_TYPE


class JsonMessage(BaseMessage):
    MESSAGE_TYPE: Final = MessageType('JsonMessage')
    data: dict

    def __init__(self, data: dict):
        self.data = data

    def serialize(self) -> bytes:
        return pickle.dumps(self.data)

    @staticmethod
    def deserialize(data: bytes, msg_type: MessageType) -> 'JsonMessage':
        return JsonMessage(pickle.loads(data))

    def get_message_type(self) -> MessageType:
        return self.MESSAGE_TYPE
