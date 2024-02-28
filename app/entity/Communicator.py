# Communicator Object
import io
import json
import logging
import pickle

import pika
import torch

from app.config import config
from app.config.logger import fed_logger

logging.getLogger("pika").setLevel(logging.ERROR)
global answer


def call_back_recv_msg(ch, method, properties, body):
    ch.stop_consuming()
    ch.close()
    global answer
    answer = body


def pack(msg, is_weight=False):
    if is_weight:
        ll = []
        for o in msg[1:]:
            to_send = io.BytesIO()
            torch.save(o, to_send, _use_new_zipfile_serialization=False)
            to_send.seek(0)
            ll.append(bytes(to_send.read()))
        return pickle.dumps(ll)
    else:
        return json.dumps(msg[1:])


def unpack(msg, is_weight=False):
    if is_weight:
        fl = []
        ll = pickle.loads(msg)
        for o in ll:
            fl.append(torch.load(io.BytesIO(o)))
        return fl
    else:
        return json.loads(msg)


class Communicator(object):
    def __init__(self):
        pass

    def open_connection(self):
        connection = pika.BlockingConnection(
            pika.URLParameters(config.mq_url))
        return connection.channel()

    def send_msg(self, exchange, msg, is_weight=False):
        bb = pack(msg, is_weight)
        channel = self.open_connection()
        channel.exchange_declare(exchange=config.cluster + "." + exchange, durable=True, exchange_type='topic')
        # channel.queue_delete(queue=config.cluster + "." + msg[0] + "." + exchange)
        channel.queue_declare(queue=config.cluster + "." + msg[0] + "." + exchange, auto_delete=True)
        channel.queue_bind(exchange=config.cluster + "." + exchange,
                           queue=config.cluster + "." + msg[0] + "." + exchange,
                           routing_key=config.cluster + "." + msg[0] + "." + exchange)
        channel.basic_publish(exchange=config.cluster + "." + exchange,
                              routing_key=config.cluster + "." + msg[0] + "." + exchange,
                              body=bb)
        channel.close()

    def recv_msg(self, exchange, expect_msg_type=None, is_weight=False):

        channel = self.open_connection()
        channel.queue_declare(queue=config.cluster + "." + expect_msg_type + "." + exchange, auto_delete=True)
        channel.queue_declare(queue=config.cluster + "." + expect_msg_type + "." + exchange, auto_delete=True)
        channel.exchange_declare(exchange=config.cluster + "." + exchange, durable=True, exchange_type='topic')
        channel.queue_bind(exchange=config.cluster + "." + exchange,
                           queue=config.cluster + "." + expect_msg_type + "." + exchange,
                           routing_key=config.cluster + "." + expect_msg_type + "." + exchange)
        # fed_logger.info('receiving message')
        # channel.basic_consume(queue=config.cluster + "." + expect_msg_type + "." + exchange,
        #                       on_message_callback=call_back_recv_msg,
        #                       auto_ack=True)
        # channel.start_consuming()
        while True:
            for method_frame, properties, body in channel.consume(
                    config.cluster + "." + expect_msg_type + "." + exchange):
                # fed_logger.info('received message')
                channel.basic_ack(method_frame.delivery_tag)
                if method_frame.delivery_tag == 1:
                    channel.stop_consuming()
                    channel.cancel()
                    channel.close()
                    msg = [expect_msg_type]
                    res = unpack(body, is_weight)
                    msg.extend(res)
                    fed_logger.info(f"{msg[0]},{type(msg[1])}")
                    return msg
        # global answer
        # while answer is None:
        #     pass
        # fed_logger.info('received done')
        # msg = [expect_msg_type]
        # res = unpack(answer, is_weight)
        # if is_weight:
        #     msg.append(res)
        # else:
        #     msg.extend(res)
        # answer = None
        # return msg
        # if channel.is_closed:
        #     msg = [expect_msg_type]
        #     res = unpack(answer, is_weight)
        #     if is_weight:
        #         msg.append(res)
        #     else:
        #         msg.extend(res)
        #     return msg
        # return ['dff', 'dfg']
