import os
import pickle
import sys
import threading
import time

from app.entity.decentralized_edge_server import FedDecentralizedEdgeServer
from app.entity.node import NodeType
from app.util import rl_utils, model_utils

sys.path.append('../../../')
from app.config import config
from app.config.logger import fed_logger
from app.entity.edge_server import FedEdgeServer
from app.entity.interface.fed_edgeserver_interface import FedEdgeServerInterface


def run_offload(server: FedEdgeServerInterface, LR, estimate_energy):
    server.initialize(config.split_layer, LR,
                      config.EDGE_NAME_TO_CLIENTS_NAME[config.EDGE_SERVER_INDEX_TO_NAME[config.index]])

    res = {}
    res['trianing_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []
    client_ips = config.EDGE_NAME_TO_CLIENTS_NAME[config.EDGE_SERVER_INDEX_TO_NAME[config.index]]
    for r in range(config.R):
        config.current_round = r
        fed_logger.info('====================================>')
        fed_logger.info('==> Round {:} Start'.format(r + 1))
        fed_logger.info("receiving global weights")
        server.get_global_weights(client_ips)
        fed_logger.info("test clients network")
        server.test_client_network(client_ips)
        fed_logger.info("sending clients network")
        server.client_network()
        fed_logger.info("test server network")
        server.test_server_network()
        fed_logger.info("receiving and sending splitting info")
        server.get_split_layers_config(client_ips)
        fed_logger.info("initializing server")
        server.initialize(server.split_layers, LR, client_ips)
        threads = {}
        fed_logger.info("start training")
        for i in range(len(client_ips)):
            threads[client_ips[i]] = threading.Thread(target=server.thread_offload_training,
                                                      args=(client_ips[i],), name=client_ips[i])
            threads[client_ips[i]].start()

        for i in range(len(client_ips)):
            threads[client_ips[i]].join()
        if estimate_energy:
            server.energy(client_ips)
        if r > 49:
            LR = config.LR * 0.1

        # energy = float(energy_estimation.energy())
        # energy /= batch_num
        # fed_logger.info(Fore.LIGHTBLUE_EX + f"Energy : {energy}" + Fore.RESET)
        fed_logger.info('==> Round {:} End'.format(r + 1))


def run_no_offload(server: FedEdgeServerInterface, LR):
    server.initialize(config.split_layer, LR,
                      config.EDGE_NAME_TO_CLIENTS_NAME[config.EDGE_SERVER_INDEX_TO_NAME[config.index]])
    res = {}
    res['trianing_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []
    client_ips = config.EDGE_NAME_TO_CLIENTS_NAME[config.EDGE_SERVER_INDEX_TO_NAME[config.index]]
    for r in range(config.R):
        config.current_round = r
        fed_logger.info('====================================>')
        fed_logger.info('==> Round {:} Start'.format(r))
        fed_logger.info("receiving global weights")
        server.no_offload_global_weights()
        # fed_logger.info("test clients network")
        # server.test_client_network(client_ips)
        # fed_logger.info("sending clients network")
        # server.client_network()
        # fed_logger.info("test server network")
        # server.test_server_network()
        threads = {}
        fed_logger.info("start training")
        for i in range(len(client_ips)):
            threads[client_ips[i]] = threading.Thread(target=server.thread_no_offload_training,
                                                      args=(client_ips[i],), name=client_ips[i])
            threads[client_ips[i]].start()

        for i in range(len(client_ips)):
            threads[client_ips[i]].join()
        if r > 49:
            LR = config.LR * 0.1


def run_decentralized_offload(server: FedDecentralizedEdgeServer, learning_rate, options: dict):
    server.initialize(learning_rate)
    iot_bw, edge_bw = [], []
    training_times = []
    rounds = []
    res = {'training_time': [], 'test_acc_record': [], 'bandwidth_record': []}
    for r in range(config.R):
        config.current_round = r
        rounds.append(r)
        fed_logger.info('====================================>')
        fed_logger.info('==> Round {:} Start'.format(r + 1))

        fed_logger.info("sending global weights")
        server.scatter_global_weights([NodeType.CLIENT])

        s_time = time.time()

        fed_logger.info("gathering neighbors network speed")
        server.gather_neighbors_network_bandwidth()

        fed_logger.info("clustering")
        server.cluster(options)

        fed_logger.info("getting state")
        state = server.get_state()
        fed_logger.info(f"STATE : {state}")
        normalized_state = []
        for bw in state:
            if r < 50:
                normalized_state.append(bw / 100_000_000)
            else:
                normalized_state.append(bw / 10_000_000)
        iot_bw.append(normalized_state[0])
        edge_bw.append(normalized_state[0])  # FIXME

        fed_logger.info("splitting")
        server.split(normalized_state, options)
        fed_logger.info(f"Split Config : {server.split_layers}")
        server.scatter_split_layers()

        if r > 49:
            learning_rate = config.LR * 0.1

        fed_logger.info("initializing server")
        server.initialize(learning_rate)

        # fed_logger.info('==> Reinitialization Finish')

        fed_logger.info("start training")
        server.start_offloading_train()

        fed_logger.info("receiving local weights")
        local_weights = server.gather_local_weights()

        fed_logger.info("aggregating weights")
        server.call_aggregation(options, local_weights)

        e_time = time.time()

        # Recording each round training time, bandwidth and test_app accuracy
        training_time = e_time - s_time
        training_times.append(training_time)

        res['training_time'].append(training_time)
        res['bandwidth_record'].append(server.bandwith())

        directory = os.path.join(config.home, 'results')
        file_path = os.path.join(directory, 'FedAdapt_res.pkl')
        os.makedirs(directory, exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(res, f)

        fed_logger.info("testing accuracy")
        test_acc = model_utils.test(server.uninet, server.testloader, server.device, server.criterion)
        res['test_acc_record'].append(test_acc)

        fed_logger.info('Round Finish')
        fed_logger.info('==> Round {:} End'.format(r + 1))
        fed_logger.info('==> Round Training Time: {:}'.format(training_time))

        rl_utils.draw_graph(10, 5, rounds, training_times, "Training time", "FL Rounds", "Training Time",
                            "/fed-flow/Graphs",
                            "trainingTime", True)
        rl_utils.draw_graph(10, 5, rounds, iot_bw, "iot BW", "FL Rounds", "iot_bw", "/fed-flow/Graphs",
                            "iot_bw", True)
        rl_utils.draw_graph(10, 5, rounds, edge_bw, "edge BW", "FL Rounds", "edge_bw", "/fed-flow/Graphs",
                            "edge_bw", True)


def run_decentralized_no_offload(server: FedEdgeServerInterface, LR):
    pass


def run(options_ins):
    LR = config.LR
    fed_logger.info('Preparing Sever.')
    offload = options_ins.get('offload')
    decentralized = options_ins.get('decentralized')
    estimate_energy = options_ins.get('energy') == "True"
    if not decentralized:
        edge_server = FedEdgeServer(
            options_ins.get('ip'), options_ins.get('port'), options_ins.get('model'),
            options_ins.get('dataset'), offload=offload)
    else:
        edge_server = FedDecentralizedEdgeServer(options_ins.get('ip'), options_ins.get('port'),
                                                 options_ins.get('model'),
                                                 options_ins.get('dataset'), offload)

    if decentralized:
        edge_server.add_neighbors(config.CURRENT_NODE_NEIGHBORS)

    fed_logger.info("start mode: " + str(options_ins.values()))
    if decentralized:
        if offload:
            run_decentralized_offload(edge_server, LR, options_ins)
        else:
            run_decentralized_no_offload(edge_server, LR)
    else:
        if offload:
            run_offload(edge_server, LR, estimate_energy)
        else:
            run_no_offload(edge_server, LR)
    # msg = edge_server.recv_msg(config.SERVER_ADDR, message_utils.finish)
    # edge_server.scatter(msg)
