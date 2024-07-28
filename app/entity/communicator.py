# Communicator Object
import logging

import pika
from kombu import Connection
from colorama import Fore

from app.config import config
from app.config.logger import fed_logger
from app.dto.message import BaseMessage, MessageType

logging.getLogger("pika").setLevel(logging.FATAL)


class Communicator(object):
    def __init__(self):
        self.connection = None
        self.channel = None
        self.url = None
        self.should_close = False
        self.send_bug = False

    @staticmethod
    def delete_all_queues():
        mq_conn = Connection(config.mq_url)
        mq_conn.connect()

        # Create a channel
        channel = mq_conn.channel()

        # Get all queues
        vhost = "/"
        manager = mq_conn.get_manager()
        queues = manager.get_queues(vhost)

        # Purge each queue
        for queue in queues:
            queue_name = queue["name"]
            channel.queue_purge(queue_name)

    def close_connection(self, ch, c):
        self.should_close = True
        if ch.is_open:
            ch.close()
        if c.is_open:
            c.close()
        while not (c.is_closed and ch.close):
            pass
        self.should_close = False

    def open_connection(self, url=None):
        fed_logger.info("connecting")
        url = config.mq_host
        self.url = url
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
            return pika.BlockingConnection(pika.ConnectionParameters(host=url, port=config.mq_port,
                                                                     credentials=pika.PlainCredentials(
                                                                         username=config.mq_user,
                                                                         password=config.mq_pass,
                                                                     )))
        except Exception as e:
            fed_logger.error(Fore.RED + f"failed to connect to rabbitmq {url}: {e}" + Fore.RESET)

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

    @staticmethod
    def declare_queue_if_not_exist(exchange_name: str, queue_name: str, routing_key: str, channel):
        queue = None
        while queue is None:
            try:
                channel.exchange_declare(exchange=exchange_name, durable=True,
                                         exchange_type='topic')
                queue = channel.queue_declare(queue=queue_name)
                channel.queue_bind(exchange=exchange_name,
                                   queue=queue_name,
                                   routing_key=routing_key)
            except Exception:
                fed_logger.error(
                    Fore.RED + f"failed to declare queue {queue_name} in exchange {exchange_name}" + Fore.RESET)
        return queue

    def send_msg(self, target_name: str, rabbitmq_endpoint: str, msg: BaseMessage):
        body = msg.serialize()
        channel, connection = self.open_connection(rabbitmq_endpoint)
        published = False
        exchange = config.cluster + "." + target_name
        queue_name = config.cluster + "." + msg.get_message_type() + "." + target_name
        routing_key = config.cluster + "." + msg.get_message_type() + "." + target_name
        self.declare_queue_if_not_exist(exchange, queue_name, routing_key, channel)
        while True:
            try:
                channel, connection = self.reconnect(connection, channel)
                fed_logger.info(
                    Fore.GREEN + f"publishing {msg.get_message_type()} to {target_name} using {rabbitmq_endpoint}" + Fore.RESET)
                channel.basic_publish(exchange=exchange,
                                      routing_key=routing_key,
                                      body=body, mandatory=True, properties=pika.BasicProperties(
                        delivery_mode=pika.DeliveryMode.Transient))
                self.close_connection(channel, connection)
                fed_logger.info(
                    Fore.GREEN + f"published {msg.get_message_type()} using {rabbitmq_endpoint}" + Fore.RESET)
                if self.send_bug:
                    fed_logger.info(
                        Fore.RED + f"published {msg.get_message_type()} using {rabbitmq_endpoint}" + Fore.RESET)
                published = True
                return

            except Exception as e:
                if published:
                    return
                self.send_bug = True
                fed_logger.error(Fore.RED + f"{e}" + Fore.RESET)
                continue

    def recv_msg(self, target_name: str, rabbitmq_endpoint: str, msg_type: MessageType) -> BaseMessage:
        channel, connection = self.open_connection(rabbitmq_endpoint)
        fed_logger.info(Fore.YELLOW + f"receiving {msg_type}" + Fore.RESET)

        while True:
            try:
                channel, connection = self.reconnect(connection, channel)
                queue = config.cluster + "." + msg_type + "." + target_name
                exchange = config.cluster + "." + target_name
                routing_key = config.cluster + "." + msg_type + "." + exchange
                self.declare_queue_if_not_exist(exchange, queue, routing_key, channel)
                fed_logger.info(
                    Fore.YELLOW + f"waiting for {msg_type} to get sent from {exchange} using {rabbitmq_endpoint}" + Fore.RESET)
                for method_frame, properties, body in channel.consume(queue=queue):
                    msg = BaseMessage.deserialize(body, msg_type)
                    channel.cancel()
                    channel.basic_ack(method_frame.delivery_tag)
                    self.close_connection(channel, connection)
                    fed_logger.info(Fore.CYAN + f"received {msg.get_message_type()} using {rabbitmq_endpoint}" + Fore.RESET)
                    return msg
            except Exception as e:
                fed_logger.info(Fore.RED + f"exception occurred while consuming {msg_type}: {e}" + Fore.RESET)
