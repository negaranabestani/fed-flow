import subprocess
import sys

import unittest

sys.path.append('../../')
from app.config import config
from app.config.logger import fed_logger


class TestFed(unittest.TestCase):
    def test_classic_1_1(self):
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

        with open("test_config/test_classic_1_1_config.py", "r") as f:
            data = f.read()

        with open("../../app/config/config.py", "a") as f:
            f.write(data)

        test = subprocess.run(['docker-compose', '-f', 'docker_compose/test_classic_1_1.yaml', 'up', '--build'])
        fed_logger.info("running client")
        if test.returncode != 0:
            self.fail(str(test.stderr))

    def test_classic_1_2(self):
        pass
