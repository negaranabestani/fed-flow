import argparse
import sys

sys.path.append('../../../')
from app.util import input_utils
from app.rl_training.flow import rl_server_flow

parser = argparse.ArgumentParser()
options = input_utils.parse_argument(parser)
rl_training_server_flow.run(options)
