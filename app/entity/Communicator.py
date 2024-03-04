# Communicator Object
import io
import json
import logging
import pickle
from retry import retry

import pika
import torch
from colorama import Fore

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


def close_connection(ch, c):
    if ch.is_open:
        ch.close()
    if c.is_open:
        c.close()


class Communicator(object):
    def __init__(self):
        pass

    def open_connection(self, url=None):
        if url is None:
            url = config.mq_host
        # url = config.mq_host

        # else:
        #     url = config.mq_host + url + ':5672/%2F'
        # url = config.mq_url
        # fed_logger.info(Fore.RED + f"{url}")
        ch = None
        c = None
        while ch is None:
            ch, c = self.connect(url)
            if c is not None and ch is None:
                c.close()
        return ch, c

    def connect(self, url):
        connection = None
        try:

            connection = pika.BlockingConnection(pika.ConnectionParameters(host=url, port=config.mq_port,
                                                                           credentials=pika.PlainCredentials(
                                                                               username=config.mq_user,
                                                                               password=config.mq_pass), heartbeat=0)
                                                 )
            return connection.channel(), connection
        except Exception:
            return None, connection

    def send_msg(self, exchange, msg, is_weight=False, url=None):

        bb = pack(msg, is_weight)
        channel, con = self.open_connection(url)
        try:
            channel.exchange_declare(exchange=config.cluster + "." + exchange, durable=True, exchange_type='topic')
            channel.queue_declare(queue=config.cluster + "." + msg[0] + "." + exchange, auto_delete=True)
            channel.queue_bind(exchange=config.cluster + "." + exchange,
                               queue=config.cluster + "." + msg[0] + "." + exchange,
                               routing_key=config.cluster + "." + msg[0] + "." + exchange)
            channel.basic_publish(exchange=config.cluster + "." + exchange,
                                  routing_key=config.cluster + "." + msg[0] + "." + exchange,
                                  body=bb)
            close_connection(channel, con)
        except Exception as e:
            fed_logger.error(e)
            close_connection(channel, con)
            self.send_msg(exchange, msg, is_weight, url)

    @retry(pika.exceptions.AMQPConnectionError, delay=5, jitter=(1, 3))
    def recv_msg(self, exchange, expect_msg_type=None, is_weight=False, url=None):
        fed_logger.info(Fore.YELLOW + f"receiving {expect_msg_type}")
        channel, con = self.open_connection(url)
        res = None
        try:
            channel.queue_declare(queue=config.cluster + "." + expect_msg_type + "." + exchange, auto_delete=True)
            channel.queue_declare(queue=config.cluster + "." + expect_msg_type + "." + exchange, auto_delete=True)
            channel.exchange_declare(exchange=config.cluster + "." + exchange, durable=True, exchange_type='topic')
            channel.queue_bind(exchange=config.cluster + "." + exchange,
                               queue=config.cluster + "." + expect_msg_type + "." + exchange,
                               routing_key=config.cluster + "." + expect_msg_type + "." + exchange)
            fed_logger.info(Fore.YELLOW + f"connected {expect_msg_type}")

            while True:
                for method_frame, properties, body in channel.consume(queue=
                                                                      config.cluster + "." + expect_msg_type + "." + exchange,
                                                                      auto_ack=True):
                    fed_logger.info(Fore.YELLOW + f"loop {expect_msg_type}")
                    # channel.basic_ack(method_frame.delivery_tag)
                    if method_frame.delivery_tag == 1:
                        msg = [expect_msg_type]
                        res = unpack(body, is_weight)
                        msg.extend(res)
                        channel.stop_consuming()
                        # channel.cancel()
                        close_connection(channel, con)
                        fed_logger.info(Fore.CYAN+f"{msg[0]},{type(msg[1])},{is_weight}")
                        return msg
        except Exception as e:
            fed_logger.info(Fore.CYAN + f"{expect_msg_type},{e},{is_weight}")
            if res is None:
                channel.stop_consuming()
                # channel.cancel()
                close_connection(channel, con)
                return self.recv_msg(exchange, expect_msg_type, is_weight, url)
            close_connection(channel, con)
            msg = [expect_msg_type]
            msg.extend(res)
            return msg
