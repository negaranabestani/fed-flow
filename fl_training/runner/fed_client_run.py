import argparse
import logging
import multiprocessing
import socket
import time

from fl_training.interface.fed_client_interface import FedClientInterface

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# ToDo use input parser to get list of input options

import sys

sys.path.append('../../')
from fl_training.entity.fed_client import Client
from config import config
from util import fl_utils

parser = argparse.ArgumentParser()
parser.add_argument('--offload', help='FedAdapt or classic FL mode', type=fl_utils.str2bool, default=False)
args = parser.parse_args()

ip_address = config.HOST2IP[socket.gethostname()]
index = config.CLIENTS_CONFIG[ip_address]
datalen = config.N / config.K
split_layer = config.split_layer[index]
LR = config.LR

logger.info('Preparing Client')
client = Client(index, ip_address, config.SERVER_ADDR, config.SERVER_PORT, datalen, 'VGG5', split_layer)

offload = args.offload


class ClientRunner:
    def run(self, client: FedClientInterface, LR):
        first = True  # First initializaiton control
        client.initialize(split_layer, offload, first, LR)
        first = False

        logger.info('Preparing Data.')
        cpu_count = multiprocessing.cpu_count()
        trainloader, classes = fl_utils.get_local_dataloader(index, cpu_count)

        if offload:
            logger.info('FedAdapt Training')
        else:
            logger.info('Classic FL Training')

        flag = False  # Bandwidth control flag.

        for r in range(config.R):
            logger.info('====================================>')
            logger.info('ROUND: {} START'.format(r))
            training_time = client.train(trainloader)
            logger.info('ROUND: {} END'.format(r))

            logger.info('==> Waiting for aggregration')
            client.upload()

            logger.info('==> Reinitialization for Round : {:}'.format(r + 1))
            s_time_rebuild = time.time()
            if offload:
                config.split_layer = client.recv_msg(client.sock)[1]

            if r > 49:
                LR = config.LR * 0.1

            client.initialize(config.split_layer[index], offload, first, LR)
            e_time_rebuild = time.time()
            logger.info('Rebuild time: ' + str(e_time_rebuild - s_time_rebuild))
            logger.info('==> Reinitialization Finish')


ClientRunner.run(client, LR)
