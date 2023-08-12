import argparse
import multiprocessing
import socket
import sys

sys.path.append('../../')
from fl_training.entity.fed_client import Client
from config import config
from config.config import *
from util import input_utils, data_utils
from config.logger import fed_logger
from fl_training.interface.fed_client_interface import FedClientInterface


class ClientRunner:
    def run_offload(self, client: FedClientInterface, LR):

        for r in range(config.R):
            fed_logger.info('====================================>')
            fed_logger.info('ROUND: {} START'.format(r))
            client.edge_global_weights()
            client.test_network()
            client.split_layer()
            client.initialize(client.split_layer, LR)
            client.offloading_train()
            client.edge_upload()
            fed_logger.info('ROUND: {} END'.format(r))

            fed_logger.info('==> Waiting for aggregration')
            if r > 49:
                LR = config.LR * 0.1

    def run_no_offload(self, client: FedClientInterface, LR):
        for r in range(config.R):
            fed_logger.info('====================================>')
            fed_logger.info('ROUND: {} START'.format(r))
            client.server_global_weights()
            client.no_offloading_train()
            client.server_upload()
            fed_logger.info('ROUND: {} END'.format(r))

            fed_logger.info('==> Waiting for aggregration')
            if r > 49:
                LR = config.LR * 0.1


parser = argparse.ArgumentParser()

options_ins = input_utils.parse_argument(parser)
ip_address = config.HOST2IP[socket.gethostname()]
index = config.CLIENTS_CONFIG[ip_address]
datalen = config.N / config.K
LR = config.LR

fed_logger.info('Preparing Client')
fed_logger.info('Preparing Data.')
cpu_count = multiprocessing.cpu_count()
indices = list(range(N))
part_tr = indices[int((N / K) * index): int((N / K) * (index + 1))]
trainloader = data_utils.get_trainloader(data_utils.get_trainset(), part_tr, cpu_count)
client_ins = Client(index, ip_address, config.SERVER_ADDR, config.SERVER_PORT, datalen, options_ins.get('model'),
                    options_ins.get('dataset'), config.split_layer[index], train_loader=trainloader)
fed_logger.info("start mode: " + str(options_ins.values()))
runner = ClientRunner()
offload = options_ins.get('offload')
if offload:
    runner.run_offload(client_ins, LR)
else:
    runner.run_no_offload(client_ins, LR)
