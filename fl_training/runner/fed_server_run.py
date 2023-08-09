import argparse
import pickle
import sys
import time

sys.path.append('../../')
from config import config
from util import model_utils, input_utils
from fl_training.entity.fed_server import FedServer
from config.logger import fed_logger
from fl_training.interface.fed_server_interface import FedServerInterface


class ServerRunner:
    def run_offload(self, server: FedServerInterface, LR, options):

        server.initialize(config.split_layer, LR)
        res = {}
        res['trianing_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []

        for r in range(config.R):
            fed_logger.info('====================================>')
            fed_logger.info('==> Round {:} Start'.format(r))

            s_time = time.time()
            server.offloading_global_weights()
            server.client_network(config.EDGE_SERVER_LIST)

            server.test_network(config.EDGE_SERVER_LIST)

            server.offloading = server.get_offloading(server.split_layers)

            server.cluster(options)

            server.state = server.concat_norm(server.ttpi(config.CLIENTS_LIST), server.offloading)

            fed_logger.info('==> Reinitialization for Round : {:}'.format(r + 1))
            server.split(options)
            server.split_layer()

            if r > 49:
                LR = config.LR * 0.1

            server.initialize(server.split_layers, LR)

            fed_logger.info('==> Reinitialization Finish')

            server.offloading_train(config.CLIENTS_LIST)
            local_weights = server.e_local_weights(config.CLIENTS_LIST)

            server.call_aggregation(options, local_weights)
            e_time = time.time()

            # Recording each round training time, bandwidth and test accuracy
            trianing_time = e_time - s_time
            res['trianing_time'].append(trianing_time)
            res['bandwidth_record'].append(server.bandwith())
            with open(config.home + '/results/FedAdapt_res.pkl', 'wb') as f:
                pickle.dump(res, f)
            test_acc = model_utils.test(server.uninet, server.testloader, server.device, server.criterion)
            res['test_acc_record'].append(test_acc)

            fed_logger.info('Round Finish')
            fed_logger.info('==> Round Training Time: {:}'.format(trianing_time))

    def run_no_offload(self, server: FedServerInterface, options):
        res = {}
        res['trianing_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []

        for r in range(config.R):
            fed_logger.info('====================================>')
            fed_logger.info('==> Round {:} Start'.format(r))

            s_time = time.time()
            server.no_offloading_gloabal_weights()
            server.cluster(options)

            server.no_offloading_train(config.CLIENTS_LIST)
            local_weights = server.c_local_weights(config.CLIENTS_LIST)

            server.call_aggregation(options, local_weights)
            e_time = time.time()

            # Recording each round training time, bandwidth and test accuracy
            trianing_time = e_time - s_time
            res['trianing_time'].append(trianing_time)
            res['bandwidth_record'].append(server.bandwith())
            with open(config.home + '/results/FedAdapt_res.pkl', 'wb') as f:
                pickle.dump(res, f)
            test_acc = model_utils.test(server.uninet, server.testloader, server.device, server.criterion)
            res['test_acc_record'].append(test_acc)

            fed_logger.info('Round Finish')
            fed_logger.info('==> Round Training Time: {:}'.format(trianing_time))


parser = argparse.ArgumentParser()
LR = config.LR

fed_logger.info('Preparing Sever.')
options_ins = input_utils.parse_argument(parser)
server_ins = FedServer(0, config.SERVER_ADDR, config.SERVER_PORT, options_ins.get('model'), options_ins.get('dataset'))
fed_logger.info("start mode: " + str(options_ins.values()))
runner = ServerRunner()
offload = options_ins.get('offload')
if offload:
    runner.run_offload(server_ins, LR, options_ins)
else:
    runner.run_no_offload(server_ins, options_ins)
