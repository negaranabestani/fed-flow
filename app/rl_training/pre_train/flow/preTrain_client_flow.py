import multiprocessing
import os
import socket
import sys
import time

sys.path.append('../../../')
from app.entity.client import Client
from app.config.config import *
from app.util import data_utils, message_utils, model_utils
from app.util.energy_estimation import *


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

    for layer in range(model_utils.get_unit_model_len()):
        for i in range(5):
            client.get_edge_global_weights()
            client.get_split_layers_config_from_edge()
            st = time.time()

            # fed_logger.info("initializing client")
            computation_start()
            client.initialize(client.split_layers, LR)

            # fed_logger.info("start training")
            client.edge_offloading_train()
            computation_end()

            # fed_logger.info("sending local weights")
            start_transmission()
            msg = client.send_local_weights_to_edge()
            end_transmission(sys.getsizeof(msg) * 8)
            et = time.time()
            tt = et - st
            comp_e, tr_e = comp_tr_energy()
            client.energy_tt(float(comp_e), tt)