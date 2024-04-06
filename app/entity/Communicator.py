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

logging.getLogger("pika").setLevel(logging.FATAL)


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
        self.send_bug = False

    def close_connection(self, ch, c):
        self.should_close = True
        if ch.is_open:
            ch.close()
        if c.is_open:
            c.close()
        while not (c.is_closed and ch.close):
            pass
        self.should_close = False
        # self.connection = None
        # self.channel = None

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
        connection = None
        while connection is None:
            connection = self.connect(url)

        channel = None
        while channel is None:
            try:
                channel = connection.channel()
                channel.basic_recover(requeue=True)
                channel.confirm_delivery()
            except Exception:
                continue
        # self.connection = self.connect(url)
        # self.connection.ioloop.start()
        fed_logger.info("connection established")
        return channel, connection

    def connect(self, url):
        try:

            # connection = pika.SelectConnection(on_close_callback=self.reconnect,
            #                                    on_open_callback=self.on_connection_open,
            #                                    parameters=pika.ConnectionParameters(host=url, port=config.mq_port,
            #                                                                         credentials=pika.PlainCredentials(
            #                                                                             username=config.mq_user,
            #                                                                             password=config.mq_pass))
            #                                    )
            return pika.BlockingConnection(pika.ConnectionParameters(host=url, port=config.mq_port,
                                                                     credentials=pika.PlainCredentials(
                                                                         username=config.mq_user,
                                                                         password=config.mq_pass)))
        except Exception as e:
            pass

    def on_connection_open(self, _unused_connection):
        fed_logger.info("opened")
        self.channel = self.connection.channel()
        fed_logger.info("connected")

    def reconnect(self, _unused_connection: pika.BlockingConnection, channel):
        if _unused_connection is not None and not self.should_close and (_unused_connection.is_closed):
            self.connection = None
            self.channel = None
            return self.open_connection(self.url)
        elif _unused_connection is not None and not (_unused_connection.is_closed) and channel.close:
            channel = _unused_connection.channel()
            channel.confirm_delivery()
            return channel, _unused_connection
        else:
            return self.open_connection(self.url)

    def send_queue(self, exchange, msg, channel):
        queue = None
        while queue is None:
            try:
                channel.exchange_declare(exchange=config.cluster + "." + exchange, durable=True,
                                         exchange_type='topic')
                queue = channel.queue_declare(queue=config.cluster + "." + msg[0] + "." + exchange)
                channel.queue_bind(exchange=config.cluster + "." + exchange,
                                   queue=config.cluster + "." + msg[0] + "." + exchange,
                                   routing_key=config.cluster + "." + msg[0] + "." + exchange)
            except Exception:
                continue
        return queue

    def send_msg(self, exchange, msg, is_weight=False, url=None):
        bb = pack(msg, is_weight)
        channel, connection = self.open_connection(url)
        queue = self.send_queue(exchange, msg, channel)
        # q2 = queue.method.message_count
        # if not msg[0].__contains__("MSG_LOCAL_ITERATION_FLAG") and q2 == 0:
        #     self.channel.basic_publish(exchange=config.cluster + "." + exchange,
        #                                routing_key=config.cluster + "." + msg[0] + "." + exchange,
        #                                body=bb, mandatory=True)
        # elif msg[0].__contains__("MSG_LOCAL_ITERATION_FLAG"):
        #     self.channel.basic_publish(exchange=config.cluster + "." + exchange,
        #                                routing_key=config.cluster + "." + msg[0] + "." + exchange,
        #                                body=bb, mandatory=True)
        while True:
            try:
                channel, connection = self.reconnect(connection, channel)
                fed_logger.info(Fore.GREEN + f"publishing {msg[0]}")
                # q2 = queue.method.message_count
                # if not msg[0].__contains__("MSG_LOCAL_ITERATION_FLAG") and q2 == 0:
                #     self.channel.basic_publish(exchange=config.cluster + "." + exchange,
                #                                routing_key=config.cluster + "." + msg[0] + "." + exchange,
                #                                body=bb, mandatory=True)
                # elif msg[0].__contains__("MSG_LOCAL_ITERATION_FLAG"):
                channel.basic_publish(exchange=config.cluster + "." + exchange,
                                      routing_key=config.cluster + "." + msg[0] + "." + exchange,
                                      body=bb, mandatory=True, properties=pika.BasicProperties(
                        delivery_mode=pika.DeliveryMode.Transient))
                # q2 = queue.method.message_count
                # fed_logger.info(Fore.MAGENTA + f" {q1},{q2}")
                self.close_connection(channel, connection)
                # if q2 > 0:
                # fed_logger.info(Fore.CYAN + f" {msg[0]},{q2}")
                fed_logger.info(Fore.GREEN + f"published {msg[0]}")
                if self.send_bug:
                    fed_logger.info(Fore.RED + f"published {msg[0]}")
                return

            except Exception as e:
                self.send_bug = True
                fed_logger.error(Fore.RED + f"{e}")
                # self.close_connection(self.channel, self.connection)
                continue
                # self.send_msg(exchange, msg, is_weight, url)

    def recv_msg(self, exchange, expect_msg_type: str = None, is_weight=False, url=None):
        channel, connection = self.open_connection(url)
        fed_logger.info(Fore.YELLOW + f"receiving {expect_msg_type}")
        res = None

        while True:
            try:
                channel, connection = self.reconnect(connection, channel)
                queue = channel.queue_declare(queue=config.cluster + "." + expect_msg_type + "." + exchange)
                channel.exchange_declare(exchange=config.cluster + "." + exchange, durable=True,
                                         exchange_type='topic')
                channel.queue_bind(exchange=config.cluster + "." + exchange,
                                   queue=config.cluster + "." + expect_msg_type + "." + exchange,
                                   routing_key=config.cluster + "." + expect_msg_type + "." + exchange)

                fed_logger.info(Fore.YELLOW + f"loop {expect_msg_type}")
                # q2 = queue.method.message_count
                # fed_logger.info(Fore.MAGENTA + f"{expect_msg_type},{q2}")

                for method_frame, properties, body in channel.consume(queue=
                                                                      config.cluster + "." + expect_msg_type + "." + exchange
                                                                      ):
                    # q2 = queue.method.message_count
                    # fed_logger.info(Fore.MAGENTA + f"{expect_msg_type},{q2}")
                    msg = [expect_msg_type]
                    res = unpack(body, is_weight)
                    msg.extend(res)
                    channel.stop_consuming()
                    channel.cancel()
                    channel.basic_ack(method_frame.delivery_tag)
                    # if not expect_msg_type.__contains__("MSG_LOCAL_ITERATION_FLAG"):
                    channel.queue_delete(queue=config.cluster + "." + expect_msg_type + "." + exchange)
                    self.close_connection(channel, connection)
                    fed_logger.info(Fore.CYAN + f"received {msg[0]},{type(msg[1])},{is_weight}")
                    return msg
            except Exception as e:
                fed_logger.info(Fore.RED + f"{expect_msg_type},{e},{is_weight}")
                fed_logger.info(Fore.RED + f"revived {expect_msg_type}")
                if res is None:
                    continue
                self.close_connection(channel, connection)
                msg = [expect_msg_type]
                msg.extend(res)
                fed_logger.info(Fore.CYAN + f"received {msg[0]},{type(msg[1])},{is_weight}")
                return msg
