# Communicator Object

import pickle
import random
import struct
import socket
import time

from app.config.logger import fed_logger


class Communicator(object):
    def __init__(self):
        self.sock = socket.socket()
        self.ip = socket.gethostbyname(socket.gethostname())

    def network_simulator(self, msg_len):
        return random.uniform(0, 0.7)*msg_len/100

    def send_msg(self, sock, msg):
        time.sleep(self.network_simulator(len(msg)))
        msg_pickle = pickle.dumps(msg)
        sock.sendall(struct.pack(">I", len(msg_pickle)))
        sock.sendall(msg_pickle)
        fed_logger.debug(msg[0] + 'sent to' + str(sock.getpeername()[0]) + ':' + str(sock.getpeername()[1]))

    def recv_msg(self, sock, expect_msg_type=None):
        msg_len = struct.unpack(">I", sock.recv(4))[0]
        msg = sock.recv(msg_len, socket.MSG_WAITALL)
        msg = pickle.loads(msg)
        fed_logger.debug(msg[0] + 'received from' + str(sock.getpeername()[0]) + ':' + str(sock.getpeername()[1]))

        if expect_msg_type is not None:
            if msg[0] == 'Finish':
                return msg
            elif msg[0] != expect_msg_type:
                raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
        return msg
