import logging
import multiprocessing
import os
import sys
import time
import warnings

from app.entity import node

sys.path.append('../../../')
from app.entity.client import Client
from app.config import config
from app.config.config import *
from app.util import data_utils, energy_estimation
from app.config.logger import fed_logger
from app.entity.interface.fed_client_interface import FedClientInterface
from colorama import Fore

warnings.filterwarnings('ignore')
logging.getLogger("requests").setLevel(logging.WARNING)


def run_edge_based(client: FedClientInterface, LR, estimate_energy):
    mx: int = int((N / K) * (index + 1))
    mn: int = int((N / K) * index)
    data_size = mx - mn
    batch_num = data_size / config.B
    # final=[]
    for r in range(config.R):
        config.current_round = r
        fed_logger.info('====================================>')
        fed_logger.info('ROUND: {} START'.format(r + 1))
        fed_logger.info("receiving global weights")
        client.get_edge_global_weights()
        st = time.time()
        fed_logger.info("test network")
        client.edge_test_network()
        fed_logger.info("receiving splitting info")
        client.get_split_layers_config_from_edge()
        fed_logger.info("initializing client")

        energy_estimation.computation_start()
        client.initialize(client.split_layers, LR)
        energy_estimation.computation_end()
        fed_logger.info("start training")
        client.edge_offloading_train()
        fed_logger.info("sending local weights")
        energy_estimation.start_transmission()
        msg = client.send_local_weights_to_edge()
        energy_estimation.end_transmission(data_utils.sizeofmessage(msg))
        et = time.time()
        fed_logger.info('ROUND: {} END'.format(r + 1))
        fed_logger.info('==> Waiting for aggregration')
        tt = et - st
        if estimate_energy:
            energy = float(energy_estimation.energy())
            # energy /= batch_num
            fed_logger.info(Fore.CYAN + f"Energy_tt : {energy}, {tt}" + Fore.RESET)
            client.energy_tt(energy, tt)
            # final.append(energy)

        if r > 49:
            LR = config.LR * 0.1
    # fed_logger.info(f"test network{final}")


def run_no_offload_edge(client: FedClientInterface, LR, estimate_energy):
    mx: int = int((N / K) * (index + 1))
    mn: int = int((N / K) * index)
    data_size = mx - mn
    batch_num = data_size / config.B

    for r in range(config.R):
        config.current_round = r
        fed_logger.info('====================================>')
        fed_logger.info('ROUND: {} START'.format(r))
        fed_logger.info("receiving global weights")
        client.get_edge_global_weights()
        st = time.time()
        fed_logger.info("start training")
        client.no_offloading_train()
        fed_logger.info("sending local weights")
        energy_estimation.start_transmission()
        msg = client.send_local_weights_to_edge()
        energy_estimation.end_transmission(data_utils.sizeofmessage(msg))
        fed_logger.info('ROUND: {} END'.format(r))
        fed_logger.info('==> Waiting for aggregration')
        if r > 49:
            LR = config.LR * 0.1
        et = time.time()
        tt = et - st
        if estimate_energy:
            energy = float(energy_estimation.energy())
            # energy /= batch_num
            fed_logger.info(Fore.CYAN + f"Energy_tt : {energy}, {tt}" + Fore.RESET)


def run_no_edge_offload(client: FedClientInterface, LR, estimate_energy):
    for r in range(config.R):
        config.current_round = r
        fed_logger.info('====================================>')
        fed_logger.info('ROUND: {} START'.format(r))
        fed_logger.info("receiving global weights")
        client.get_server_global_weights()
        st = time.time()
        fed_logger.info("test_app network")
        energy_estimation.start_transmission()
        msg = client.test_network()
        energy_estimation.end_transmission(data_utils.sizeofmessage(msg))
        fed_logger.info("receiving splitting info")
        client.get_split_layers_config()
        energy_estimation.computation_start()
        fed_logger.info("initializing client")
        client.initialize(client.split_layers, LR)
        fed_logger.info("start training")
        energy_estimation.computation_end()
        client.offloading_train()
        fed_logger.info("sending local weights")
        energy_estimation.start_transmission()
        msg = client.send_local_weights_to_server()
        energy_estimation.end_transmission(data_utils.sizeofmessage(msg))
        fed_logger.info('ROUND: {} END'.format(r))
        fed_logger.info('==> Waiting for aggregration')
        if r > 49:
            LR = config.LR * 0.1
        et = time.time()
        tt = et - st
        if estimate_energy:
            energy = float(energy_estimation.energy())
            # energy /= batch_num
            fed_logger.info(Fore.CYAN + f"Energy_tt : {energy}, {tt}" + Fore.RESET)


def run_no_edge(client: FedClientInterface, LR, estimate_energy):
    for r in range(config.R):
        config.current_round = r
        fed_logger.info('====================================>')
        fed_logger.info('ROUND: {} START'.format(r))
        fed_logger.info("receiving global weights")

        st = time.time()

        client.get_server_global_weights()
        fed_logger.info("start training")
        client.no_offloading_train()
        fed_logger.info("sending local weights")
        client.send_local_weights_to_server()

        tt = time.time()
        fed_logger.info('ROUND: {} END'.format(r))

        fed_logger.info('==> Waiting for aggregration')
        if r > 49:
            LR = config.LR * 0.1
        if estimate_energy:
            energy = float(energy_estimation.energy())
            fed_logger.info(Fore.CYAN + f"Energy_tt : {energy}, {tt - st}" + Fore.RESET)


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
    trainloader = data_utils.get_trainloader(data_utils.get_trainset(), part_tr, 0)

    estimate_energy = False
    offload = options_ins.get('offload')
    edge_based = options_ins.get('edgebased')
    if options_ins.get("energy") == "True":
        energy_estimation.init(os.getpid())
        estimate_energy = True

    ip = options_ins.get('ip')
    port = options_ins.get('port')

    node.start_server_thread(ip, port)

    if edge_based and offload:

        client_ins = Client(ip=ip, port=port,
                            server=config.CLIENT_NAME_TO_EDGE_NAME[config.CLIENTS_INDEX_TO_NAME[index]],
                            datalen=datalen,
                            model_name=options_ins.get('model'),
                            dataset=options_ins.get('dataset'), train_loader=trainloader, LR=LR, edge_based=edge_based,
                            )
        run_edge_based(client_ins, LR, estimate_energy)
    elif edge_based and not offload:
        client_ins = Client(ip=ip, port=port,
                            server=config.CLIENT_NAME_TO_EDGE_NAME[config.CLIENTS_INDEX_TO_NAME[index]],
                            datalen=datalen, model_name=options_ins.get('model'),
                            dataset=options_ins.get('dataset'), train_loader=trainloader, LR=LR, edge_based=edge_based,
                            )
        run_no_offload_edge(client_ins, LR, estimate_energy)
    elif offload:
        client_ins = Client(ip=ip, port=port, server=config.SERVER_ADDR,
                            datalen=datalen, model_name=options_ins.get('model'),
                            dataset=options_ins.get('dataset'), train_loader=trainloader, LR=LR, edge_based=edge_based,
                            )
        run_no_edge_offload(client_ins, LR, estimate_energy)
    else:
        client_ins = Client(ip=ip, port=port, server=config.SERVER_ADDR,
                            datalen=datalen, model_name=options_ins.get('model'),
                            dataset=options_ins.get('dataset'), train_loader=trainloader, LR=LR, edge_based=edge_based,
                            )
        run_no_edge(client_ins, LR, estimate_energy)
    # client_ins.recv_msg(config.CLIENTS_INDEX[index], message_utils.finish)

# parser = argparse.ArgumentParser()
# options = input_utils.parse_argument(parser)
# run(options)
