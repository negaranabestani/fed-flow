import argparse
import sys

from app.entity.communicator import Communicator

sys.path.append('../../../')
from app.util import input_utils
from app.fl_training.flow import fed_edgeserver_flow

parser = argparse.ArgumentParser()
options = input_utils.parse_argument(parser)
Communicator.purge_all_queues()
fed_edgeserver_flow.run(options)
