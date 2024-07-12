import argparse
import sys

from app.entity.communicator import Communicator

sys.path.append('../../../')
from app.util import input_utils
from app.fl_training.flow import fed_client_flow

parser = argparse.ArgumentParser()
options = input_utils.parse_argument(parser)
Communicator.delete_all_queues()
fed_client_flow.run(options)
