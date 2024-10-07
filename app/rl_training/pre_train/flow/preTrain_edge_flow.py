import os
import socket
import sys
import threading

import numpy as np
from colorama import Fore

from app.util import model_utils

sys.path.append('../../../../')
from app.config import config
from app.entity.edge_server import FedEdgeServer
from app.util import energy_estimation
from app.config.logger import fed_logger


def run(options_ins):
    LR = config.learning_rate
    ip_address = socket.gethostname()
    # fed_logger.info('Preparing Sever.')
    edge_server = FedEdgeServer(options_ins.get('model'), options_ins.get('dataset'),
                                offload=options_ins.get('offload'))
    # fed_logger.info("start mode: " + str(options_ins.values()))

    compEnergyOfLayers = np.zeros((model_utils.get_unit_model_len(), 10))
    compTTOfLayers = np.zeros((model_utils.get_unit_model_len(), 10))

    edge_server.initialize(config.split_layer, LR, config.EDGE_NAME_TO_CLIENTS_NAME[config.EDGE_SERVER_INDEX_TO_NAME[config.index]])
    client_ips = config.EDGE_NAME_TO_CLIENTS_NAME[config.EDGE_SERVER_INDEX_TO_NAME[config.index]]
    energy_estimation.init(os.getpid())

    for layer in range(model_utils.get_unit_model_len() - 1):
        for j in range(10):
            # fed_logger.info('====================================>')
            # fed_logger.info('==> Round {:} Start'.format(r))
            #
            # fed_logger.info("receiving global weights")
            edge_server.get_global_weights(client_ips)

            # fed_logger.info("test clients network")
            # server.test_client_network(client_ips)

            # fed_logger.info("sending clients network")
            # server.client_network()

            # fed_logger.info("test server network")
            # server.test_server_network()

            # fed_logger.info("receiving and sending splitting info")
            edge_server.get_split_layers_config(client_ips)

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

            comp_e, tr_e, comp_time, tr_time = energy_estimation.energy_and_time_comp_tr()

            fed_logger.info(Fore.LIGHTBLUE_EX + f"Computation Energy : {comp_e}")
            compEnergyOfLayers[layer + 1][j] = comp_e
            compTTOfLayers[layer+1][j] = comp_time
    fed_logger.info("=======================================================")
    fed_logger.info("Pre Train Ended.")
    fed_logger.info(f"Energy consumption of layer 0 to {model_utils.get_unit_model_len()} is:")
    fed_logger.info(compEnergyOfLayers)
    fed_logger.info(f"Training time of layer 0 to {model_utils.get_unit_model_len()} is:")
    fed_logger.info(compTTOfLayers)
