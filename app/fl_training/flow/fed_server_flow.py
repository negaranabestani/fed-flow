import os
import pickle
import sys
import time

from app.dto.message import JsonMessage

sys.path.append('../../../')
from app.config import config
from app.util import model_utils
from app.entity.server import FedServer
from app.config.logger import fed_logger
from app.entity.interface.fed_server_interface import FedServerInterface
from app.util import rl_utils


def run_edge_based_no_offload(server: FedServerInterface, LR, options):
    res = {}
    res['training_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []

    for r in range(config.R):
        config.current_round = r
        fed_logger.info('====================================>')
        fed_logger.info('==> Round {:} Start'.format(r))

        fed_logger.info("sending global weights")
        server.edge_offloading_global_weights()
        s_time = time.time()
        fed_logger.info("clustering")
        server.cluster(options)
        fed_logger.info("receiving local weights")
        local_weights = server.e_local_weights(config.CLIENTS_LIST)

        fed_logger.info("prepare aggregation local weights")
        local_weight_list = server.prepare_aggregation_local_weights(config.CLIENTS_LIST, local_weights)
        fed_logger.info("aggregating weights")
        server.aggregator.aggregate(options.get('aggregation'), local_weight_list)

        e_time = time.time()

        # Recording each round training time, bandwidth and test_app accuracy
        training_time = e_time - s_time
        fed_logger.info("testing accuracy")
        test_acc = model_utils.test(server.uninet, server.testloader, server.device, server.criterion)
        res['test_acc_record'].append(test_acc)

        fed_logger.info('Round Finish')
        fed_logger.info('==> Round Training Time: {:}'.format(training_time))


def run_edge_based_offload(server: FedServerInterface, LR, options, estimate_energy):
    server.initialize(config.split_layer, LR)
    training_time = 0
    energy_tt_list = []
    # all_splitting = rl_utils.allPossibleSplitting(7, 1)
    energy_x = []
    training_y = []
    # split_list = [[[0, 1]], [[1, 2]], [[2, 3]], [[3, 4]], [[4, 5]], [[5, 6]]]
    avgEnergy, tt = [], []
    iotBW, edgeBW = [], []
    x = []
    for c in config.CLIENTS_LIST:
        energy_tt_list.append([0, 0])
    res = {}
    res['training_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []
    fed_logger.info(f"OPTION: {options}")
    for r in range(config.R):
        config.current_round = r
        x.append(r)
        fed_logger.info('====================================>')
        fed_logger.info('==> Round {:} Start'.format(r + 1))

        fed_logger.info("sending global weights")
        server.edge_offloading_global_weights()

        s_time = time.time()

        fed_logger.info("receiving client network info")
        server.client_network(config.EDGE_SERVER_LIST)

        fed_logger.info("test edge servers network")
        server.test_network(config.EDGE_SERVER_LIST)

        fed_logger.info("preparing state...")
        server.offloading = server.get_offloading(server.split_layers)

        fed_logger.info("clustering")
        server.cluster(options)

        fed_logger.info("getting state")
        offloading = server.split_layers

        state = server.edge_based_state()
        fed_logger.info(f"STATE : {state}")
        fed_logger.info("state: " + str(state))
        normalizedState = []
        for bw in state:
            if r < 50:
                normalizedState.append(bw / 100_000_000)
            else:
                normalizedState.append(bw / 10_000_000)
        iotBW.append(normalizedState[0])
        edgeBW.append(normalizedState[1])

        fed_logger.info("splitting")
        server.split(normalizedState, options)
        fed_logger.info(f"Agent Action : {server.split_layers}")
        # server.split_layers = split_list[r]
        server.send_split_layers_config_to_edges()

        if r > 49:
            LR = config.LR * 0.1

        fed_logger.info("initializing server")
        server.initialize(server.split_layers, LR)

        # fed_logger.info('==> Reinitialization Finish')

        fed_logger.info("start training")
        server.edge_offloading_train(config.CLIENTS_LIST)

        fed_logger.info("receiving local weights")
        local_weights = server.e_local_weights(config.CLIENTS_LIST)

        fed_logger.info("prepare aggregation local weights")
        local_weight_list = server.prepare_aggregation_local_weights(config.CLIENTS_LIST, local_weights)
        fed_logger.info("aggregating weights")
        server.aggregator.aggregate(options.get('aggregation'), local_weight_list)

        if estimate_energy:
            energy_tt_list = server.e_energy_tt(config.CLIENTS_LIST)
            print(f"E TT :{energy_tt_list}")
            clientEnergy = []
            for i in range(config.K):
                clientEnergy.append(energy_tt_list[i][0])
            avgEnergy.append(sum(clientEnergy) / int(config.K))

        e_time = time.time()

        # Recording each round training time, bandwidth and test_app accuracy
        training_time = e_time - s_time
        tt.append(training_time)

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

        rl_utils.draw_graph(10, 5, x, tt, "Training time", "FL Rounds", "Training Time", "/fed-flow/Graphs",
                            "trainingTime", True)
        if estimate_energy:
            rl_utils.draw_graph(10, 5, x, avgEnergy, "Energy time", "FL Rounds", "Energy", "/fed-flow/Graphs",
                                "energy", True)
        rl_utils.draw_graph(10, 5, x, iotBW, "iot BW", "FL Rounds", "iotBW", "/fed-flow/Graphs",
                            "iotBW", True)
        rl_utils.draw_graph(10, 5, x, edgeBW, "edge BW", "FL Rounds", "edgeBW", "/fed-flow/Graphs",
                            "edgeBW", True)


def run_no_edge_offload(server: FedServerInterface, LR, options):
    server.initialize(config.split_layer, LR)
    res = {}
    res['training_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []

    for r in range(config.R):
        config.current_round = r

        fed_logger.info('====================================>')
        fed_logger.info('==> Round {:} Start'.format(r))

        fed_logger.info("sending global weights")
        server.no_offloading_global_weights()
        s_time = time.time()
        fed_logger.info("test clients network")
        server.test_network(config.CLIENTS_LIST)

        fed_logger.info("preparing state...")
        server.offloading = server.get_offloading(server.split_layers)

        fed_logger.info("clustering")
        server.cluster(options)
        fed_logger.info("getting state")
        ttpi = server.ttpi(config.CLIENTS_LIST)
        state = server.concat_norm(ttpi, server.offloading)

        fed_logger.info("splitting")
        server.split(state, options)
        server.send_split_layers_config()
        fed_logger.info("initializing server")
        server.initialize(server.split_layers, LR)

        fed_logger.info("start training")
        server.no_edge_offloading_train(config.CLIENTS_LIST)
        fed_logger.info("receiving local weights")
        local_weights = server.c_local_weights(config.CLIENTS_LIST)

        fed_logger.info("prepare aggregation local weights")
        local_weight_list = server.prepare_aggregation_local_weights(config.CLIENTS_LIST, local_weights)
        fed_logger.info("aggregating weights")
        server.aggregator.aggregate(options.get('aggregation'), local_weight_list)

        e_time = time.time()

        # Recording each round training time, bandwidth and test accuracy
        training_time = e_time - s_time
        res['training_time'].append(training_time)
        res['bandwidth_record'].append(server.bandwith())

        directory = os.path.join(config.home, 'results')
        file_path = os.path.join(directory, 'FedAdapt_res.pkl')
        os.makedirs(directory, exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(res, f)

        test_acc = model_utils.test(server.uninet, server.testloader, server.device, server.criterion)
        res['test_acc_record'].append(test_acc)

        fed_logger.info('Round Finish')
        fed_logger.info('==> Round Training Time: {:}'.format(training_time))


def run_no_edge(server: FedServerInterface, options):
    res = {}
    res['training_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []

    for r in range(config.R):
        config.current_round = r

        fed_logger.info('====================================>')
        fed_logger.info('==> Round {:} Start'.format(r))

        fed_logger.info("sending global weights")
        server.no_offloading_global_weights()
        s_time = time.time()
        server.cluster(options)
        fed_logger.info("start training")
        server.no_offloading_train(config.CLIENTS_LIST)
        fed_logger.info("receiving local weights")
        local_weights = server.c_local_weights(config.CLIENTS_LIST)

        fed_logger.info("prepare aggregation local weights")
        local_weight_list = server.prepare_aggregation_local_weights(config.CLIENTS_LIST, local_weights)
        fed_logger.info("aggregating weights")
        server.aggregator.aggregate(options.get('aggregation'), local_weight_list)

        e_time = time.time()

        # Recording each round training time, bandwidth and test accuracy
        training_time = e_time - s_time
        res['training_time'].append(training_time)
        res['bandwidth_record'].append(server.bandwith())

        directory = os.path.join(config.home, 'results')
        file_path = os.path.join(directory, 'FedAdapt_res.pkl')
        os.makedirs(directory, exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(res, f)

        test_acc = model_utils.test(server.uninet, server.testloader, server.device, server.criterion)
        res['test_acc_record'].append(test_acc)

        fed_logger.info('Round Finish')
        fed_logger.info('==> Round Training Time: {:}'.format(training_time))


def run(options_ins):
    LR = config.LR
    fed_logger.info('Preparing Sever.')
    fed_logger.info("start mode: " + str(options_ins.values()))
    offload = options_ins.get('offload')
    edge_based = options_ins.get('edgebased')
    estimate_energy = False
    if options_ins.get('energy') == "True":
        estimate_energy = True
    if edge_based and offload:
        server_ins = FedServer(options_ins.get('ip'), options_ins.get('port'), options_ins.get('model'),
                               options_ins.get('dataset'), offload, edge_based)
        run_edge_based_offload(server_ins, LR, options_ins, estimate_energy)
    elif edge_based and not offload:
        server_ins = FedServer(options_ins.get('ip'), options_ins.get('port'), options_ins.get('model'),
                               options_ins.get('dataset'), offload, edge_based)
        run_edge_based_no_offload(server_ins, LR, options_ins)
    elif offload and not edge_based:
        server_ins = FedServer(options_ins.get('ip'), options_ins.get('port'), options_ins.get('model'),
                               options_ins.get('dataset'), offload, edge_based)
        run_no_edge_offload(server_ins, LR, options_ins)
    else:
        server_ins = FedServer(options_ins.get('ip'), options_ins.get('port'), options_ins.get('model'),
                               options_ins.get('dataset'), offload, edge_based)
        run_no_edge(server_ins, options_ins)
