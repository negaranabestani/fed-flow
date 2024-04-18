import socket
import sys
import threading
import os

from app.util import model_utils
from colorama import Fore
import numpy as np
sys.path.append('../../../../')
from app.config import config
from app.entity.edge_server import FedEdgeServer
from app.util import message_utils, energy_estimation
from app.config.logger import fed_logger


def run(options_ins):
    LR = config.LR
    ip_address = socket.gethostname()
    # fed_logger.info('Preparing Sever.')
    edge_server = FedEdgeServer(options_ins.get('model'), options_ins.get('dataset'),
                                offload=options_ins.get('offload'))
    # fed_logger.info("start mode: " + str(options_ins.values()))

    energyOfLayers = np.zeros((model_utils.get_unit_model_len()))

    edge_server.initialize(config.split_layer, LR, config.EDGE_MAP[config.EDGE_SERVER_CONFIG[config.index]])
    client_ips = config.EDGE_MAP[config.EDGE_SERVER_CONFIG[config.index]]
    energy_estimation.init(os.getpid())

    for layer in range(model_utils.get_unit_model_len()-1):
        for i in range(10):
            # fed_logger.info('====================================>')
            # fed_logger.info('==> Round {:} Start'.format(r))
            #
            # fed_logger.info("receiving global weights")
            edge_server.global_weights(client_ips)

            # fed_logger.info("test clients network")
            # server.test_client_network(client_ips)

            # fed_logger.info("sending clients network")
            # server.client_network()

            # fed_logger.info("test server network")
            # server.test_server_network()

            # fed_logger.info("receiving and sending splitting info")
            edge_server.split_layer(client_ips)

            # fed_logger.info("initializing server")
            edge_server.initialize(edge_server.split_layers, LR, client_ips)
            threads = {}

            # fed_logger.info("start training")
            for i in range(len(client_ips)):
                threads[client_ips[i]] = threading.Thread(target=edge_server.thread_offload_training,
                                                          args=(client_ips[i],), name=client_ips[i])
                threads[client_ips[i]].start()

            for i in range(len(client_ips)):
                threads[client_ips[i]].join()
            edge_server.energy(client_ips)
            energy = float(energy_estimation.energy())
            # energy /= batch_num
            fed_logger.info(Fore.LIGHTBLUE_EX + f"Energy : {energy}")
            energyOfLayers[layer+1] += energy
        energyOfLayers[layer+1] /= 10
    fed_logger.info("=======================================================")
    fed_logger.info("Pre Train Ended.")
    fed_logger.info(f"Energy consumption of layer 0 to {model_utils.get_unit_model_len()} is:")
    fed_logger.info(energyOfLayers)
