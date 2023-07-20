import pickle
import socket
import struct
from enum import Enum

from config.logger import fed_logger


class Message(Enum):
    test_network = 'MSG_TEST_NETWORK',
    initial_global_weights_server_to_client = 'MSG_INITIAL_GLOBAL_WEIGHTS_SERVER_TO_CLIENT',
    training_time_per_iteration = 'MSG_TRAINING_TIME_PER_ITERATION',
    local_activations = 'MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER',
    server_gradients = 'MSG_SERVER_GRADIENTS_SERVER_TO_CLIENT_',
    local_weights = 'MSG_LOCAL_WEIGHTS_CLIENT_TO_SERVER',
    split_layers = 'SPLIT_LAYERS'


def send_msg(sock, msg):
    msg_pickle = pickle.dumps(msg)
    sock.sendall(struct.pack(">I", len(msg_pickle)))
    sock.sendall(msg_pickle)
    fed_logger.debug(msg[0] + 'sent to' + str(sock.getpeername()[0]) + ':' + str(sock.getpeername()[1]))


def recv_msg(sock, expect_msg_type=None):
    msg_len = struct.unpack(">I", sock.recv(4))[0]
    msg = sock.recv(msg_len, socket.MSG_WAITALL)
    msg = pickle.loads(msg)
    fed_logger.debug(msg[0] + 'received from' + str(sock.getpeername()[0]) + ':' + str(sock.getpeername()[1]))

    if (expect_msg_type is not None) and (msg[0] != expect_msg_type):
        raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
    return msg
