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
    def run(self, server: FedServerInterface, LR, first, options, offload):
        first = False

        res = {}
        res['trianing_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []

        for r in range(config.R):
            fed_logger.info('====================================>')
            fed_logger.info('==> Round {:} Start'.format(r))

            s_time = time.time()
            server.global_weights()

            server.client_network(config.EDGE_SERVER_LIST)

            server.test_network(config.EDGE_SERVER_LIST)

            server.offloading = server.get_offloading(server.split_layers)

            server.cluster(options)

            server.state = server.concat_norm(server.ttpi(config.CLIENTS_LIST), server.offloading)

            fed_logger.info('==> Reinitialization for Round : {:}'.format(r + 1))
            server.split(options)
            if r > 49:
                LR = config.LR * 0.1
            server.initialize(server.split_layers, offload, first, LR)

            fed_logger.info('==> Reinitialization Finish')
            if offload:
                server.offloading_train(config.CLIENTS_LIST)
            else:
                server.no_offloading_train(config.CLIENTS_LIST)

            eweights = server.local_weights(config.CLIENTS_LIST)

            server.call_aggregation(options, eweights)
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
# parser.add_argument('--offload', help='FedAdapt or classic FL mode', type=fl_utils.str2bool, default=False)
# args = parser.parse_args()
LR = config.LR
# offload = args.offload
first = True  # First initializaiton control

fed_logger.info('Preparing Sever.')
options_ins = input_utils.parse_argument(parser)
server_ins = FedServer(0, config.SERVER_ADDR, config.SERVER_PORT, options_ins.get('model'), options_ins.get('dataset'))
fed_logger.info("start mode: " + str(options_ins.values()))
runner = ServerRunner()
runner.run(server_ins, LR, first, options_ins, options_ins.get('offload'))
