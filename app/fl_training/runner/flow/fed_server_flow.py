import pickle
import socket
import sys
import time

sys.path.append('../../../../')
from app.config import config
from app.util import model_utils, message_utils
from app.fl_training.entity.fed_server import FedServer
from app.config.logger import fed_logger
from app.fl_training.interface.fed_server_interface import FedServerInterface


def run_edge_based(server: FedServerInterface, LR, options):
    server.initialize(config.split_layer, LR)
    training_time = 0
    res = {}
    res['training_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []

    for r in range(config.R):
        fed_logger.info('====================================>')
        fed_logger.info('==> Round {:} Start'.format(r))

        s_time = time.time()
        fed_logger.info("sending global weights")
        server.edge_offloading_global_weights()
        # fed_logger.info("receiving client network info")
        # server.client_network(config.EDGE_SERVER_LIST)
        #
        # fed_logger.info("test edge servers network")
        # server.test_network(config.EDGE_SERVER_LIST)

        fed_logger.info("preparing state...")
        server.offloading = server.get_offloading(server.split_layers)

        fed_logger.info("clustering")
        server.cluster(options)
        fed_logger.info("getting state")
        offloading = server.split_layers
        state = server.edge_based_state(training_time, offloading)

        fed_logger.info("splitting")
        server.split(state, options)
        server.split_layer()

        if r > 49:
            LR = config.LR * 0.1

        fed_logger.info("initializing server")
        server.initialize(server.split_layers, LR)

        # fed_logger.info('==> Reinitialization Finish')

        fed_logger.info("start training")
        server.edge_offloading_train(config.CLIENTS_LIST)
        fed_logger.info("receiving local weights")
        local_weights = server.e_local_weights(config.CLIENTS_LIST)
        fed_logger.info("aggregating weights")
        server.call_aggregation(options, local_weights)
        e_time = time.time()

        # Recording each round training time, bandwidth and test_app accuracy
        training_time = e_time - s_time
        res['training_time'].append(training_time)
        res['bandwidth_record'].append(server.bandwith())
        with open(config.home + '/results/FedAdapt_res.pkl', 'wb') as f:
            pickle.dump(res, f)

        fed_logger.info("testing accuracy")
        test_acc = model_utils.test(server.uninet, server.testloader, server.device, server.criterion)
        res['test_acc_record'].append(test_acc)

        fed_logger.info('Round Finish')
        fed_logger.info('==> Round Training Time: {:}'.format(training_time))


def run_no_edge_offload(server: FedServerInterface, LR, options):
    server.initialize(config.split_layer, LR)
    res = {}
    res['training_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []

    for r in range(config.R):
        res = {}
        res['training_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []

        for r in range(config.R):
            fed_logger.info('====================================>')
            fed_logger.info('==> Round {:} Start'.format(r))

            s_time = time.time()
            fed_logger.info("sending global weights")
            server.no_offloading_global_weights()
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
            server.split_layer()
            fed_logger.info("initializing server")
            server.initialize(server.split_layers, LR)

            fed_logger.info("start training")
            server.no_edge_offloading_train(config.CLIENTS_LIST)
            fed_logger.info("receiving local weights")
            local_weights = server.c_local_weights(config.CLIENTS_LIST)
            fed_logger.info("aggregating weights")
            server.call_aggregation(options, local_weights)
            e_time = time.time()

            # Recording each round training time, bandwidth and test accuracy
            training_time = e_time - s_time
            res['training_time'].append(training_time)
            res['bandwidth_record'].append(server.bandwith())
            with open(config.home + '/results/FedAdapt_res.pkl', 'wb') as f:
                pickle.dump(res, f)
            test_acc = model_utils.test(server.uninet, server.testloader, server.device, server.criterion)
            res['test_acc_record'].append(test_acc)

            fed_logger.info('Round Finish')
            fed_logger.info('==> Round Training Time: {:}'.format(training_time))


def run_no_edge(server: FedServerInterface, options):
    res = {}
    res['training_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []

    for r in range(config.R):
        fed_logger.info('====================================>')
        fed_logger.info('==> Round {:} Start'.format(r))

        s_time = time.time()
        fed_logger.info("sending global weights")
        server.no_offloading_global_weights()
        server.cluster(options)
        fed_logger.info("start training")
        server.no_offloading_train(config.CLIENTS_LIST)
        fed_logger.info("receiving local weights")
        local_weights = server.c_local_weights(config.CLIENTS_LIST)
        fed_logger.info("aggregating weights")
        server.call_aggregation(options, local_weights)
        e_time = time.time()

        # Recording each round training time, bandwidth and test accuracy
        training_time = e_time - s_time
        res['training_time'].append(training_time)
        res['bandwidth_record'].append(server.bandwith())
        with open(config.home + '/results/FedAdapt_res.pkl', 'wb') as f:
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
    if edge_based:
        server_ins = FedServer(config.SERVER_ADDR, config.SERVER_PORT, options_ins.get('model'),
                               options_ins.get('dataset'), offload, edge_based)
        run_edge_based(server_ins, LR, options_ins)
    elif offload:
        server_ins = FedServer(config.SERVER_ADDR, config.SERVER_PORT, options_ins.get('model'),
                               options_ins.get('dataset'), offload, edge_based)
        run_no_edge_offload(server_ins, LR, options_ins)
    else:
        server_ins = FedServer(config.SERVER_ADDR, config.SERVER_PORT, options_ins.get('model'),
                               options_ins.get('dataset'), offload, edge_based)
        run_no_edge(server_ins, options_ins)
    msg = [message_utils.finish, True]
    server_ins.scatter(msg)
