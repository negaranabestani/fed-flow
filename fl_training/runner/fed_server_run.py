import argparse
import pickle
import sys
import time

from config.logger import fed_logger
from fl_method import splitting
from fl_training.interface.fed_server_interface import FedServerInterface

sys.path.append('../../')
from config import config
from util import fl_utils
from fl_training.entity.fed_server import FedServer


class ServerRunner:
    def run(self, server: FedServerInterface, LR, first):
        server.initialize(config.split_layer, offload, first, LR)
        first = False

        if offload:
            fed_logger.info('FedAdapt Training')
        else:
            fed_logger.info('Classic FL Training')

        res = {}
        res['trianing_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []

        for r in range(config.R):
            fed_logger.info('====================================>')
            fed_logger.info('==> Round {:} Start'.format(r))

            s_time = time.time()
            state, bandwidth = server.train(thread_number=config.K, client_ips=config.CLIENTS_LIST)
            server.aggregate(config.CLIENTS_LIST)
            e_time = time.time()

            # Recording each round training time, bandwidth and test accuracy
            trianing_time = e_time - s_time
            res['trianing_time'].append(trianing_time)
            res['bandwidth_record'].append(bandwidth)

            test_acc = fl_utils.test(server.uninet, server.testloader, server.device, server.criterion)
            res['test_acc_record'].append(test_acc)

            with open(config.home + '/results/FedAdapt_res.pkl', 'wb') as f:
                pickle.dump(res, f)

            fed_logger.info('Round Finish')
            fed_logger.info('==> Round Training Time: {:}'.format(trianing_time))

            fed_logger.info('==> Reinitialization for Round : {:}'.format(r + 1))
            if offload:
                split_layers = splitting.rl_splitting(state, server.group_labels)
                fed_logger.info('Next Round OPs: ' + str(config.split_layer))
                msg = ['SPLIT_LAYERS', config.split_layer]
                server.scatter(msg)
            else:
                split_layers = config.split_layer

            if r > 49:
                LR = config.LR * 0.1

            server.initialize(split_layers, offload, first, LR)
            fed_logger.info('==> Reinitialization Finish')


parser = argparse.ArgumentParser()
parser.add_argument('--offload', help='FedAdapt or classic FL mode', type=fl_utils.str2bool, default=False)
args = parser.parse_args()
# ToDo use input parser to get list of input options
LR = config.LR
offload = args.offload
first = True  # First initializaiton control

fed_logger.info('Preparing Sever.')
server_ins = FedServer(0, config.SERVER_ADDR, config.SERVER_PORT, 'VGG5')
runner = ServerRunner()
runner.run(server_ins, LR, first)
