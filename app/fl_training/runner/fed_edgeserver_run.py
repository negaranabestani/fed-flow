import argparse
import sys

sys.path.append('../../../')
from app.util import input_utils
from app.fl_training.flow import fed_edgeserver_flow

parser = argparse.ArgumentParser()
options = input_utils.parse_argument(parser)
fed_edgeserver_flow.run(options)
