import multiprocessing
import os
import socket
import sys
import time

sys.path.append('../../../')
from app.entity.client import Client
from app.config.config import *
from app.util import data_utils, message_utils
from app.config.logger import fed_logger
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
                    dataset=options_ins.get('dataset'), train_loader=trainloader, LR=LR, edge_based=edge_based,
                    )

    preTrain(client)

    for r in range(config.max_episodes):
        # fed_logger.info('====================================>')
        # fed_logger.info('Episode: {} START'.format(r))

        # fed_logger.info("receiving global weights")
        client.edge_global_weights()

        # fed_logger.info("test network")
        # client.test_network()

        # fed_logger.info("receiving splitting info")
        client.split_layer()
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
        # fed_logger.info('ROUND: {} END'.format(r))
        # fed_logger.info('==> Waiting for aggregration')

        # fed_logger.info(f"Energy : {enery}")

        client.energy_tt(float(energy()) / batch_num, tt)

        for i in range(config.max_timesteps):
            # fed_logger.info('====================================>')
            # fed_logger.info('TimeStep: {} START'.format(r))

            # fed_logger.info("receiving global weights")
            client.edge_global_weights()

            # fed_logger.info("test network")
            # client.test_network()

            # fed_logger.info("receiving splitting info")
            client.split_layer()
            st = time.time()

            # fed_logger.info("initializing client")
            computation_start()
            client.initialize(client.split_layers, LR)
            computation_end()

            # fed_logger.info("start training")
            client.edge_offloading_train()

            # fed_logger.info("sending local weights")
            start_transmission()
            msg = client.edge_upload()
            et = time.time()
            tt = et - st
            end_transmission(sys.getsizeof(msg) * 8)

            # fed_logger.info('ROUND: {} END'.format(r))
            # fed_logger.info('==> Waiting for aggregration')

            # fed_logger.info(f"Energy : {enery}")
            client.energy_tt(float(energy()) / batch_num, tt)

            if r > 49:
                LR = config.LR * 0.1
    msg = client.recv_msg(client.sock, message_utils.finish)


# parser = argparse.ArgumentParser()
# options = input_utils.parse_argument(parser)
# run(options)
def preTrain(client):
    # splittingLayer = rl_utils.allPossibleSplitting(modelLen=config.model_len, deviceNumber=1)
    mx: int = int((N / K) * (index + 1))
    mn: int = int((N / K) * index)
    data_size = mx - mn
    batch_num = data_size / config.B

    for i in range(5):
        fed_logger.info("Getting params...")
        client.edge_global_weights()
        fed_logger.info("test network")
        # client.test_network()
        client.split_layer()
        st = time.time()

        computation_start()
        client.initialize(client.split_layers, 0.1)
        fed_logger.info("Initialized ...")
        computation_end()

        fed_logger.info("start training on edge....")
        client.edge_offloading_train()

        start_transmission()
        fed_logger.info("Uploaded")
        msg = client.edge_upload()
        et = time.time()
        tt = et - st
        end_transmission(sys.getsizeof(msg) * 8)
        client.energy_tt(float(energy()) / batch_num, tt)
        fed_logger.info(f"{float(energy()) / batch_num}, {tt}")
