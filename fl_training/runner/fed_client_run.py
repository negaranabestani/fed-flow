import argparse
import multiprocessing
import socket
import sys
import time

from config.logger import fed_logger
from fl_training.interface.fed_client_interface import FedClientInterface

# ToDo use input parser to get list of input options

sys.path.append('../../')
from fl_training.entity.fed_client import Client
from config import config
from util import fl_utils, input_utils


class ClientRunner:
    def run(self, client: FedClientInterface, LR, offload):
        first = True  # First initializaiton control
        client.initialize(client.split_layer, offload, first, LR)
        first = False

        fed_logger.info('Preparing Data.')
        cpu_count = multiprocessing.cpu_count()
        trainloader, classes = fl_utils.get_local_dataloader(index, cpu_count)

        if offload:
            fed_logger.info('FedAdapt Training')
        else:
            fed_logger.info('Classic FL Training')

        flag = False  # Bandwidth control flag.

        for r in range(config.R):
            fed_logger.info('====================================>')
            fed_logger.info('ROUND: {} START'.format(r))
            training_time = client.train(trainloader)
            fed_logger.info('ROUND: {} END'.format(r))

            fed_logger.info('==> Waiting for aggregration')
            client.upload()

            fed_logger.info('==> Reinitialization for Round : {:}'.format(r + 1))
            s_time_rebuild = time.time()
            if offload:
                config.split_layer = client.recv_msg(client.sock)[1]

            if r > 49:
                LR = config.LR * 0.1

            client.initialize(config.split_layer[index], offload, first, LR)
            e_time_rebuild = time.time()
            fed_logger.info('Rebuild time: ' + str(e_time_rebuild - s_time_rebuild))
            fed_logger.info('==> Reinitialization Finish')


parser = argparse.ArgumentParser()
# parser.add_argument('--offload', help='FedAdapt or classic FL mode', type=fl_utils.str2bool, default=False)
# args = parser.parse_args()
options_ins = input_utils.parse_argument(parser)
ip_address = config.HOST2IP[socket.gethostname()]
index = config.CLIENTS_CONFIG[ip_address]
datalen = config.N / config.K
# split_layer = config.split_layer[index]
LR = config.LR

fed_logger.info('Preparing Client')
client_ins = Client(index, ip_address, config.SERVER_ADDR, config.SERVER_PORT, datalen, options_ins.get('model'),
                    options_ins.get('dataset'), config.split_layer[index])

runner = ClientRunner()
runner.run(client_ins, LR, options_ins.get('offload'))
