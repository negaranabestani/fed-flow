import multiprocessing
import os
import socket
import sys
import time

from colorama import Fore

sys.path.append('../../../')
from app.entity.client import Client
from app.config.config import *
from app.util import data_utils, model_utils
from app.util.energy_estimation import *
from app.config.logger import fed_logger


def run(options_ins):
    ip_address = socket.gethostname()
    #  fed_logger.info("start mode: " + str(options_ins.values()))

    index = config.index
    datalen = config.N / config.K
    LR = config.LR
    mx: int = int((N / K) * (index + 1))
    mn: int = int((N / K) * index)
    data_size = mx - mn
    batch_num = data_size / config.B

    # fed_logger.info('Preparing Client')
    # fed_logger.info('Preparing Data.')
    cpu_count = multiprocessing.cpu_count()
    indices = list(range(N))
    part_tr = indices[int((N / K) * index): int((N / K) * (index + 1))]
    trainloader = data_utils.get_trainloader(data_utils.get_trainset(), part_tr, cpu_count)

    offload = options_ins.get('offload')
    edge_based = options_ins.get('edgebased')

    init(os.getpid())
    client = Client(server=config.CLIENT_MAP[config.CLIENTS_INDEX[index]],
                    datalen=datalen, model_name=options_ins.get('model'),
                    dataset=options_ins.get('dataset'), train_loader=trainloader, LR=LR, edge_based=edge_based)

    energyOfLayers0 = 0

    for layer in range(model_utils.get_unit_model_len() - 1):
        for i in range(10):
            client.edge_global_weights()
            client.edge_split_layer()
            st = time.time()

            # fed_logger.info("initializing client")
            computation_start()
            client.initialize(client.split_layers, LR)

            # fed_logger.info("start training")
            client.edge_offloading_train()
            computation_end()

            # fed_logger.info("sending local weights")
            start_transmission()
            msg = client.edge_upload()
            end_transmission(sys.getsizeof(msg) * 8)
            et = time.time()
            tt = et - st
            comp_e, tr_e = comp_tr_energy()
            client.energy_tt(float(comp_e), tt)
            if layer == 0:
                energyOfLayers0 += float(comp_e)
    energyOfLayers0 /= 10
    fed_logger.info(Fore.RED + f"Energy of Layer 0: {energyOfLayers0}")
