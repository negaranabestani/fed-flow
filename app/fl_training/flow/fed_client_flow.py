import logging
import multiprocessing
import sys
import time
import warnings

from app.entity.decentralized_client import DecentralizedClient
from app.util.mobility_data_utils import start_mobility_simulation_thread

sys.path.append('../../../')
from app.entity.client import Client
from app.config import config
from app.config.config import *
from app.util import data_utils, energy_estimation
from app.config.logger import fed_logger
from colorama import Fore

warnings.filterwarnings('ignore')
logging.getLogger("requests").setLevel(logging.WARNING)


def run_edge_based(client: Client, LR, estimate_energy):
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


def run_no_offload_edge(client: Client, LR, estimate_energy):
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


def run_no_edge_offload(client: Client, LR, estimate_energy):
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


def run_no_edge(client: Client, LR, estimate_energy):
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


def run_offload_decentralized(client: DecentralizedClient, learning_rate):
    for r in range(config.R):
        config.current_round = r
        fed_logger.info('====================================>')
        fed_logger.info('ROUND: {} START'.format(r + 1))
        fed_logger.info("receiving global weights")
        client.gather_global_weights()
        fed_logger.info("test network")
        client.scatter_network_speed_to_edges()
        fed_logger.info("receiving splitting info")
        client.gather_split_config()
        fed_logger.info("initializing client")

        client.initialize(client.split_layers, learning_rate)
        fed_logger.info("start training")
        client.start_offloading_train()
        fed_logger.info("sending local weights")
        client.scatter_local_weights()
        fed_logger.info('ROUND: {} END'.format(r + 1))

        if r > 49:
            learning_rate = config.LR * 0.1


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

    estimate_energy = options_ins.get("energy") == "True"
    offload = options_ins.get('offload')
    edge_based = options_ins.get('edgebased')
    decentralized = options_ins.get('decentralized')
    mobility = options_ins.get('mobility')

    if estimate_energy:
        energy_estimation.init(os.getpid())

    ip = options_ins.get('ip')
    port = options_ins.get('port')

    client = None
    if decentralized:
        client = DecentralizedClient(ip=ip, port=port, model_name=options_ins.get('model'),
                                     dataset=options_ins.get('dataset'), train_loader=trainloader, LR=LR)
        if mobility:
            start_mobility_simulation_thread(client)

    elif edge_based:
        client = Client(ip=ip, port=port,
                        model_name=options_ins.get('model'),
                        dataset=options_ins.get('dataset'), train_loader=trainloader, LR=LR, edge_based=edge_based,
                        )
    else:
        client = Client(ip=ip, port=port, model_name=options_ins.get('model'),
                        dataset=options_ins.get('dataset'), train_loader=trainloader, LR=LR, edge_based=edge_based,
                        )

    if decentralized:
        client.add_neighbors(config.CURRENT_NODE_NEIGHBORS)

    if decentralized and offload:
        run_offload_decentralized(client, LR)
    elif edge_based and offload:
        run_edge_based(client, LR, estimate_energy)
    elif edge_based and not offload:
        run_no_offload_edge(client, LR, estimate_energy)
    elif offload:
        run_no_edge_offload(client, LR, estimate_energy)
    else:
        run_no_edge(client, LR, estimate_energy)
