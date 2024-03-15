# Communicator Object
import io
import json
import logging
import pickle

import pika
import torch
from colorama import Fore

from app.config import config
from app.config.logger import fed_logger

logging.getLogger("pika").setLevel(logging.ERROR)


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
        self.connection = None
        self.channel = None
        self.url = None
        self.should_close = False

    def close_connection(self, ch, c):
        self.should_close = True
        if ch.is_open:
            ch.close()
        if c.is_open:
            c.close()
        while not (c.is_closed and ch.close):
            pass
        self.should_close = False
        self.connection = None
        self.channel = None

    def open_connection(self, url=None):
        fed_logger.info("connecting")
        # if url is None:
        #     url = config.mq_host
        url = config.mq_host
        self.url = url

        # url = config.mq_host

        # else:
        #     url = config.mq_host + url + ':5672/%2F'
        # url = config.mq_url
        # fed_logger.info(Fore.RED + f"{url}")
        while self.connection is None:
            self.connect(url)

        while self.channel is None:
            self.channel = self.connection.channel()
            self.channel.basic_recover(requeue=True)
        # self.connection = self.connect(url)
        # self.connection.ioloop.start()
        fed_logger.info("started")

    def connect(self, url):
        try:

            # connection = pika.SelectConnection(on_close_callback=self.reconnect,
            #                                    on_open_callback=self.on_connection_open,
            #                                    parameters=pika.ConnectionParameters(host=url, port=config.mq_port,
            #                                                                         credentials=pika.PlainCredentials(
            #                                                                             username=config.mq_user,
            #                                                                             password=config.mq_pass))
            #                                    )
            self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=url, port=config.mq_port,
                                                                                credentials=pika.PlainCredentials(
                                                                                    username=config.mq_user,
                                                                                    password=config.mq_pass)))


        except Exception as e:
            pass

    def on_connection_open(self, _unused_connection):
        fed_logger.info("opened")
        self.channel = self.connection.channel()
        fed_logger.info("connected")

    def reconnect(self, _unused_connection: pika.BlockingConnection):
        if not self.should_close and (_unused_connection.is_closed):
            self.connection = None
            self.channel = None
            self.open_connection(self.url)

    def send_msg(self, exchange, msg, is_weight=False, url=None):

        bb = pack(msg, is_weight)
        self.open_connection(url)
        fed_logger.info(Fore.YELLOW + f"published {msg[0]}")
        try:
            self.channel.exchange_declare(exchange=config.cluster + "." + exchange, durable=True, exchange_type='topic')
            self.channel.queue_declare(queue=config.cluster + "." + msg[0] + "." + exchange)
            self.channel.queue_bind(exchange=config.cluster + "." + exchange,
                                    queue=config.cluster + "." + msg[0] + "." + exchange,
                                    routing_key=config.cluster + "." + msg[0] + "." + exchange)
            self.channel.basic_publish(exchange=config.cluster + "." + exchange,
                                       routing_key=config.cluster + "." + msg[0] + "." + exchange,
                                       body=bb)
            self.close_connection(self.channel, self.connection)
        except Exception as e:
            # fed_logger.error(e)
            self.close_connection(self.channel, self.connection)
            self.send_msg(exchange, msg, is_weight, url)

    # @retry(pika.exceptions.AMQPConnectionError, delay=5, jitter=(1, 3))
    def recv_msg(self, exchange, expect_msg_type=None, is_weight=False, url=None):
        self.open_connection(url)
        fed_logger.info(Fore.YELLOW + f"receiving {expect_msg_type}")
        res = None
        try:
            self.channel.queue_declare(queue=config.cluster + "." + expect_msg_type + "." + exchange)
            self.channel.exchange_declare(exchange=config.cluster + "." + exchange, durable=True, exchange_type='topic')
            self.channel.queue_bind(exchange=config.cluster + "." + exchange,
                                    queue=config.cluster + "." + expect_msg_type + "." + exchange,
                                    routing_key=config.cluster + "." + expect_msg_type + "." + exchange)
            fed_logger.info(Fore.YELLOW + f"connected {expect_msg_type}")

            while True:
                try:
                    self.reconnect(self.connection)
                    self.channel.queue_declare(queue=config.cluster + "." + expect_msg_type + "." + exchange)
                    self.channel.exchange_declare(exchange=config.cluster + "." + exchange, durable=True,
                                                  exchange_type='topic')
                    self.channel.queue_bind(exchange=config.cluster + "." + exchange,
                                            queue=config.cluster + "." + expect_msg_type + "." + exchange,
                                            routing_key=config.cluster + "." + expect_msg_type + "." + exchange)

                    fed_logger.info(Fore.YELLOW + f"loop {expect_msg_type}")
                    for method_frame, properties, body in self.channel.consume(queue=
                                                                               config.cluster + "." + expect_msg_type + "." + exchange
                                                                               ):
                        msg = [expect_msg_type]
                        res = unpack(body, is_weight)
                        msg.extend(res)
                        self.channel.stop_consuming()
                        # self.channel.cancel()
                        self.channel.basic_ack(method_frame.delivery_tag)
                        self.channel.queue_delete(queue=config.cluster + "." + expect_msg_type + "." + exchange)
                        self.close_connection(self.channel, self.connection)
                        fed_logger.info(Fore.CYAN + f"received {msg[0]},{type(msg[1])},{is_weight}")
                        return msg
                except Exception as e:
                    fed_logger.info(Fore.RED + f"{expect_msg_type},{e},{is_weight}")
                    fed_logger.info(Fore.RED + f"revived {expect_msg_type}")
                    if res is None:
                        continue
                    self.close_connection(self.channel, self.connection)
                    msg = [expect_msg_type]
                    msg.extend(res)
                    fed_logger.info(Fore.CYAN + f"received {msg[0]},{type(msg[1])},{is_weight}")
                    return msg
        except Exception as e:
            fed_logger.info(Fore.CYAN + f"{expect_msg_type},{e},{is_weight}")
            if res is None:
                # self.channel.stop_consuming()
                # self.channel.cancel()
                self.close_connection(self.channel, self.connection)
                return self.recv_msg(exchange, expect_msg_type, is_weight, url)
            self.close_connection(self.channel, self.connection)
            msg = [expect_msg_type]
            msg.extend(res)
            return msg
