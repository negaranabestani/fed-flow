import subprocess
import sys
from multiprocessing import Process

sys.path.append('../../')
import unittest

from app.fl_training.runner.flow import fed_server_flow, fed_client_flow
from app.config import config
from app.config.logger import fed_logger


def test_classic_1_1_flow():
    options = {
        'aggregation': 'fed_avg',
        'clustering': 'none_clustering',
        'splitting': 'fake_splitting',
        'model': 'vgg',
        'dataset': 'cifar10',
        'offload': False,
        'datasetlink': '',
        'modellink': ''
    }
    config.dataset_name = options.get('dataset')
    config.model_name = options.get('model')
    # try:
    #     subprocess.run(['python3', '../../app/fl_training/runner/fed_server_flow.py'])
    # except subprocess.CalledProcessError as e:
    #     fed_logger.log("server failed: " + str(e))
    #
    # server = subprocess.Popen(['python3', '../../app/fl_training/runner/fed_server_flow.py'], stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    # fed_logger.info("running server")
    # o, r = server.communicate()
    # if server.returncode != 0:
    #     raise Exception("server failed: " + str(r))
    # client = subprocess.Popen(['python3', '../../app/fl_training/runner/fed_client_flow.py'])
    # fed_logger.info("running client")
    # o, r = client.communicate()
    # if client.returncode != 0:
    #     raise Exception("client failed" + str(r))
    # server.wait()
    # client.wait()
    server = Process(target=fed_server_flow.run, args=(options,))
    server.start()
    client = Process(target=fed_client_flow.run, args=(options,))
    client.start()
    client.join()
    server.join()

    if server.exitcode > 0:
        raise Exception("server failed")
    if client.exitcode > 0:
        raise Exception("client failed")

    # commands = ['command1', 'command2']
    # procs = [Popen(i) for i in commands]
    # for p in procs:
    #     p.wait()


def test_classic_1_2_flow():
    options = {
        'aggregation': 'fed_avg',
        'clustering': 'none_clustering',
        'splitting': 'fake_splitting',
        'model': 'vgg',
        'dataset': 'cifar10',
        'offload': False,
        'datasetlink': '',
        'modellink': ''
    }
    config.dataset_name = options.get('dataset')
    config.model_name = options.get('model')

    server = Process(target=fed_server_flow.run, args=(options,))
    server.start()
    client1 = Process(target=fed_client_flow.run, args=(options,))
    client1.start()
    client2 = Process(target=fed_client_flow.run, args=(options,))
    client2.start()

    client2.join()
    client1.join()
    server.join()

    if server.exitcode > 0:
        raise Exception("server failed")
    if client1.exitcode > 0:
        raise Exception("client1 failed")
    if client2.exitcode > 0:
        raise Exception("client2 failed")


class TestFed(unittest.TestCase):
    def test_classic_1_1(self):
        test = subprocess.run(['docker-compose', '-f', 'docker-compose/test_classic_1_1.yaml', 'up'])
        fed_logger.info("running client")
        if test.returncode != 0:
            self.fail(str(test.stderr))

    def test_classic_1_2(self):
        try:
            test_classic_1_2_flow()
        except Exception as e:
            self.fail(str(e))
