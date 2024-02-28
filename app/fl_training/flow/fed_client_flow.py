import logging
import multiprocessing
import os
import socket
import sys
import time
import warnings

sys.path.append('../../../')
from app.entity.client import Client
from app.config import config
from app.config.config import *
from app.util import data_utils, message_utils, energy_estimation
from app.config.logger import fed_logger
from app.entity.interface.fed_client_interface import FedClientInterface
from colorama import Fore

warnings.filterwarnings('ignore')
logging.getLogger("requests").setLevel(logging.WARNING)


def run_edge_based(client: FedClientInterface, LR):
    mx: int = int((N / K) * (index + 1))
    mn: int = int((N / K) * index)
    data_size = mx - mn
    batch_num = data_size / config.B

    for r in range(config.R):
        fed_logger.info('====================================>')
        fed_logger.info('ROUND: {} START'.format(r))

        fed_logger.info("receiving global weights")
        client.edge_global_weights()
        st = time.time()
        # fed_logger.info("test network")
        # client.test_network()
        fed_logger.info("receiving splitting info")
        client.split_layer()
        fed_logger.info("initializing client")

        energy_estimation.computation_start()
        client.initialize(client.split_layers, LR)
        energy_estimation.computation_end()
        fed_logger.info("start training")
        client.edge_offloading_train()
        fed_logger.info("sending local weights")
        energy_estimation.start_transmission()
        msg = client.edge_upload()
        energy_estimation.end_transmission(data_utils.sizeofmessage(msg))
        et = time.time()
        fed_logger.info('ROUND: {} END'.format(r))
        fed_logger.info('==> Waiting for aggregration')
        tt = et - st
        energy = float(energy_estimation.energy())
        # energy /= batch_num
        fed_logger.info(Fore.CYAN + f"Energy_tt : {energy}, {tt}")
        client.energy_tt(energy, tt)

        if r > 49:
            LR = config.LR * 0.1


def run_no_offload_edge(client: FedClientInterface, LR):
    mx: int = int((N / K) * (index + 1))
    mn: int = int((N / K) * index)
    data_size = mx - mn
    batch_num = data_size / config.B

    for r in range(config.R):
        fed_logger.info('====================================>')
        fed_logger.info('ROUND: {} START'.format(r))
        fed_logger.info("receiving global weights")
        client.edge_global_weights()
        st = time.time()
        fed_logger.info("start training")
        client.no_offloading_train()
        fed_logger.info("sending local weights")
        energy_estimation.start_transmission()
        msg = client.edge_upload()
        energy_estimation.end_transmission(data_utils.sizeofmessage(msg))
        fed_logger.info('ROUND: {} END'.format(r))
        fed_logger.info('==> Waiting for aggregration')
        if r > 49:
            LR = config.LR * 0.1
        et = time.time()
        tt = et - st
        energy = float(energy_estimation.energy())
        # energy /= batch_num
        fed_logger.info(Fore.CYAN + f"Energy_tt : {energy}, {tt}")


def run_no_edge_offload(client: FedClientInterface, LR):
    for r in range(config.R):

        fed_logger.info('====================================>')
        fed_logger.info('ROUND: {} START'.format(r))
        fed_logger.info("receiving global weights")
        client.server_global_weights()
        st = time.time()
        fed_logger.info("test_app network")
        energy_estimation.start_transmission()
        msg = client.test_network()
        energy_estimation.end_transmission(data_utils.sizeofmessage(msg))
        fed_logger.info("receiving splitting info")
        client.split_layer()
        energy_estimation.computation_start()
        fed_logger.info("initializing client")
        client.initialize(client.split_layers, LR)
        fed_logger.info("start training")
        energy_estimation.computation_end()
        client.offloading_train()
        fed_logger.info("sending local weights")
        energy_estimation.start_transmission()
        msg = client.server_upload()
        energy_estimation.end_transmission(data_utils.sizeofmessage(msg))
        fed_logger.info('ROUND: {} END'.format(r))
        fed_logger.info('==> Waiting for aggregration')
        if r > 49:
            LR = config.LR * 0.1
        et = time.time()
        tt = et - st
        energy = float(energy_estimation.energy())
        # energy /= batch_num
        fed_logger.info(Fore.CYAN + f"Energy_tt : {energy}, {tt}")


def run_no_edge(client: FedClientInterface, LR):
    for r in range(config.R):
        fed_logger.info('====================================>')
        fed_logger.info('ROUND: {} START'.format(r))
        fed_logger.info("receiving global weights")
        client.server_global_weights()
        fed_logger.info("start training")
        client.no_offloading_train()
        fed_logger.info("sending local weights")
        client.server_upload()
        fed_logger.info('ROUND: {} END'.format(r))

        fed_logger.info('==> Waiting for aggregration')
        if r > 49:
            LR = config.LR * 0.1


def run(options_ins):
    fed_logger.info("start mode: " + str(options_ins.values()))
    index = config.index
    datalen = config.N / config.K
    LR = config.LR

    fed_logger.info('Preparing Client')
    fed_logger.info('Preparing Data.')
    cpu_count = multiprocessing.cpu_count()
    indices = list(range(N))
    part_tr = indices[int((N / K) * index): int((N / K) * (index + 1))]
    trainloader = data_utils.get_trainloader(data_utils.get_trainset(), part_tr, cpu_count)

    offload = options_ins.get('offload')
    edge_based = options_ins.get('edgebased')
    if edge_based and offload:
        energy_estimation.init(os.getpid())
        client_ins = Client(server=config.CLIENT_MAP[config.CLIENTS_INDEX[index]], datalen=datalen,
                            model_name=options_ins.get('model'),
                            dataset=options_ins.get('dataset'), train_loader=trainloader, LR=LR, edge_based=edge_based,
                            )
        run_edge_based(client_ins, LR)
    elif edge_based and not offload:
        energy_estimation.init(os.getpid())
        client_ins = Client(server=config.CLIENT_MAP[config.CLIENTS_INDEX[index]],
                            datalen=datalen, model_name=options_ins.get('model'),
                            dataset=options_ins.get('dataset'), train_loader=trainloader, LR=LR, edge_based=edge_based,
                            )
        run_no_offload_edge(client_ins, LR)
    elif offload:
        energy_estimation.init(os.getpid())
        client_ins = Client(server=config.SERVER_ADDR,
                            datalen=datalen, model_name=options_ins.get('model'),
                            dataset=options_ins.get('dataset'), train_loader=trainloader, LR=LR, edge_based=edge_based,
                            )
        run_no_edge_offload(client_ins, LR)
    else:
        energy_estimation.init(os.getpid())
        client_ins = Client(server=config.SERVER_ADDR,
                            datalen=datalen, model_name=options_ins.get('model'),
                            dataset=options_ins.get('dataset'), train_loader=trainloader, LR=LR, edge_based=edge_based,
                            )
        run_no_edge(client_ins, LR)
    client_ins.recv_msg(message_utils.finish)

# parser = argparse.ArgumentParser()
# options = input_utils.parse_argument(parser)
# run(options)
