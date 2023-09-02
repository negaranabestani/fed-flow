import subprocess
import sys

import unittest

sys.path.append('../../')
from app.config.logger import fed_logger


class TestFed(unittest.TestCase):
    def test_classic_1_1(self):
        with open("test_config/test_classic_1_1_config.py", "r") as f:
            data = f.read()

        with open("../../app/config/config.py", "w") as f:
            f.write(data)

        test = subprocess.run(['docker-compose', '-f', 'docker_compose/test_classic_1_1.yaml', 'up', '--build'])
        fed_logger.info("running client")
        if test.returncode != 0:
            self.fail(str(test.stderr))

    def test_classic_1_3(self):
        with open("test_config/test_classic_1_3_config.py", "r") as f:
            data = f.read()

        with open("../../app/config/config.py", "w") as f:
            f.write(data)
        test = subprocess.run(['docker-compose', '-f', 'docker_compose/test_classic_1_3.yaml', 'up', '--build'])
        fed_logger.info(str(test.stdout))
        if test.returncode != 0:
            self.fail(str(test.stderr))

    def test_offloading_1_1(self):
        with open("test_config/test_offloading_1_1_config.py", "r") as f:
            data = f.read()

        with open("../../app/config/config.py", "w") as f:
            f.write(data)
        test = subprocess.run(
            ['docker-compose', '-f', 'docker_compose/test_offloading_1_1.yaml', 'up', '--build'])
        fed_logger.info(str(test.stdout))
        if test.returncode != 0:
            self.fail(str(test.stderr))

    def test_fake_offloading_1_1_1(self):
        with open("test_config/test_fake_offloading_1_1_1_config.py", "r") as f:
            data = f.read()

        with open("../../app/config/config.py", "w") as f:
            f.write(data)
        test = subprocess.run(
            ['docker-compose', '-f', 'docker_compose/test_fake_offloading_1_1_1.yaml', 'up', '--build'])
        fed_logger.info(str(test.stdout))
        if test.returncode != 0:
            self.fail(str(test.stderr))

    def test_no_offloading_1_1_1(self):
        with open("test_config/test_no_offloading_1_1_1_config.py", "r") as f:
            data = f.read()

        with open("../../app/config/config.py", "w") as f:
            f.write(data)
        test = subprocess.run(
            ['docker-compose', '-f', 'docker_compose/test_no_offloading_1_1_1.yaml', 'up', '--build'])
        fed_logger.info(str(test.stdout))
        if test.returncode != 0:
            self.fail(str(test.stderr))

