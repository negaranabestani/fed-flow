import argparse
import pickle
import sys
import time

sys.path.append('../../../')
from app.config import config
from app.util import input_utils, model_utils
from app.fl_training.entity.fed_server import FedServer
from app.config.logger import fed_logger
from app.fl_training.interface.fed_server_interface import FedServerInterface


class ServerRunner:
    def run_offload(self, server: FedServerInterface, LR, options):

        server.initialize(config.split_layer, LR)
        res = {}
        res['trianing_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []

        for r in range(config.R):
            fed_logger.info('====================================>')
            fed_logger.info('==> Round {:} Start'.format(r))

            s_time = time.time()
            fed_logger.info("sending global weights")
            server.offloading_global_weights()
            fed_logger.info("receiving client network info")
            server.client_network(config.EDGE_SERVER_LIST)

            fed_logger.info("test edge servers network")
            server.test_network(config.EDGE_SERVER_LIST)

            fed_logger.info("preparing state...")
            server.offloading = server.get_offloading(server.split_layers)

            fed_logger.info("clustering")
            server.cluster(options)
            fed_logger.info("getting state")
            ttpi = server.ttpi(config.CLIENTS_LIST)
            server.state = server.concat_norm(ttpi, server.offloading)

            fed_logger.info("splitting")
            server.split(options)
            server.split_layer()

            if r > 49:
                LR = config.LR * 0.1

            fed_logger.info("initializing server")
            server.initialize(server.split_layers, LR)

            # fed_logger.info('==> Reinitialization Finish')

            fed_logger.info("start training")
            server.offloading_train(config.CLIENTS_LIST)
            fed_logger.info("receiving local weights")
            local_weights = server.e_local_weights(config.CLIENTS_LIST)
            fed_logger.info("aggregating weights")
            server.call_aggregation(options, local_weights)
            e_time = time.time()

            # Recording each round training time, bandwidth and test accuracy
            trianing_time = e_time - s_time
            res['trianing_time'].append(trianing_time)
            res['bandwidth_record'].append(server.bandwith())
            with open(config.home + '/results/FedAdapt_res.pkl', 'wb') as f:
                pickle.dump(res, f)

            fed_logger.info("testing accuracy")
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
            server.no_offloading_global_weights()
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
fed_logger.info("start mode: " + str(options_ins.values()))
runner = ServerRunner()
offload = options_ins.get('offload')
if offload:
    server_ins = FedServer(0, config.SERVER_ADDR, config.SERVER_PORT, options_ins.get('model'),
                           options_ins.get('dataset'), config.EDGE_SERVER_LIST)
    runner.run_offload(server_ins, LR, options_ins)
else:
    server_ins = FedServer(0, config.SERVER_ADDR, config.SERVER_PORT, options_ins.get('model'),
                           options_ins.get('dataset'), config.CLIENTS_LIST)
    runner.run_no_offload(server_ins, options_ins)
