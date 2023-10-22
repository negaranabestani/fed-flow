import multiprocessing
import socket
import sys
import pyRAPL

sys.path.append('../../../')
from app.entity.client import Client
from app.config import config
from app.config.config import *
from app.util import data_utils, message_utils, rl_utils
from app.config.logger import fed_logger


def run(options_ins):
    ip_address = socket.gethostname()
    #  fed_logger.info("start mode: " + str(options_ins.values()))

    index = config.index
    datalen = config.N / config.K
    LR = config.LR

    # fed_logger.info('Preparing Client')
    # fed_logger.info('Preparing Data.')
    cpu_count = multiprocessing.cpu_count()
    indices = list(range(N))
    part_tr = indices[int((N / K) * index): int((N / K) * (index + 1))]
    trainloader = data_utils.get_trainloader(data_utils.get_trainset(), part_tr, cpu_count)

    offload = options_ins.get('offload')
    edge_based = options_ins.get('edgebased')

    client = Client(server_addr=config.CLIENT_MAP[ip_address],
                    server_port=config.EDGESERVER_PORT[config.CLIENT_MAP[ip_address]],
                    datalen=datalen, model_name=options_ins.get('model'),
                    dataset=options_ins.get('dataset'), train_loader=trainloader, LR=LR, edge_based=edge_based)

    pyRAPL.setup()

    preTrain(client)
    meter = pyRAPL.Measurement('bar')

    for r in range(config.max_episodes):
        # fed_logger.info('====================================>')
        # fed_logger.info('Episode: {} START'.format(r))

        meter.begin()

        # fed_logger.info("receiving global weights")
        client.edge_global_weights()

        # fed_logger.info("test network")
        # client.test_network()

        # fed_logger.info("receiving splitting info")
        client.split_layer()

        # fed_logger.info("initializing client")
        client.initialize(client.split_layers, LR)

        # fed_logger.info("start training")
        client.edge_offloading_train()

        # fed_logger.info("sending local weights")
        client.edge_upload()

        # fed_logger.info('ROUND: {} END'.format(r))
        # fed_logger.info('==> Waiting for aggregration')

        meter.end()
        enery = 0
        if meter.result.pkg != None:
            for en in meter.result.pkg:
                enery += en

        # fed_logger.info(f"Energy : {enery}")

        client.energy(enery)

        for i in range(config.max_timesteps):
            # fed_logger.info('====================================>')
            # fed_logger.info('TimeStep: {} START'.format(r))

            meter.begin()

            # fed_logger.info("receiving global weights")
            client.edge_global_weights()

            # fed_logger.info("test network")
            # client.test_network()

            # fed_logger.info("receiving splitting info")
            client.split_layer()

            # fed_logger.info("initializing client")
            client.initialize(client.split_layers, LR)

            # fed_logger.info("start training")
            client.edge_offloading_train()

            # fed_logger.info("sending local weights")
            client.edge_upload()

            # fed_logger.info('ROUND: {} END'.format(r))
            # fed_logger.info('==> Waiting for aggregration')

            meter.end()

            enery = 0
            if meter.result.pkg != None:
                for en in meter.result.pkg:
                    enery += en

            # fed_logger.info(f"Energy : {enery}")
            client.energy(enery)

            if r > 49:
                LR = config.LR * 0.1
    msg = client.recv_msg(client.sock, message_utils.finish)


# parser = argparse.ArgumentParser()
# options = input_utils.parse_argument(parser)
# run(options)
def preTrain(client):
    splittingLayer = rl_utils.allPossibleSplitting(modelLen=config.model_len, deviceNumber=1)

    meter = pyRAPL.Measurement('bar')

    for splitting in splittingLayer:
        meter.begin()
        client.edge_global_weights()
        # fed_logger.info("test network")
        # client.test_network()
        client.split_layer()
        client.initialize(client.split_layers, 0.1)
        client.edge_offloading_train()
        client.edge_upload()

        meter.end()

        enery = 0

        if meter.result.pkg != None:
            for en in meter.result.pkg:
                enery += en
        client.energy(enery)

    meter.begin()
    client.edge_global_weights()
    # fed_logger.info("test network")
    # client.test_network()
    client.split_layer()
    client.initialize(client.split_layers, 0.1)
    client.edge_offloading_train()
    client.edge_upload()

    meter.end()

    enery = 0

    if meter.result.pkg != None:
        for en in meter.result.pkg:
            enery += en
    client.energy(enery)
