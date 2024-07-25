import argparse
import sys

sys.path.append('../../../')
from app.config import config
from app.entity.communicator import Communicator
from app.util import input_utils
from app.fl_training.flow import fed_client_flow

parser = argparse.ArgumentParser()
options = input_utils.parse_argument(parser)
if hasattr(config, 'DEBUG') and config.DEBUG:
    Communicator.delete_all_queues()
fed_client_flow.run(options)
