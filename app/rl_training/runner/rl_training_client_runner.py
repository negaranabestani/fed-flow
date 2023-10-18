import argparse
import sys

sys.path.append('../../../')
from app.util import input_utils
from app.rl_training.runner.flow import rl_training_client_flow

parser = argparse.ArgumentParser()
options = input_utils.parse_argument(parser)
rl_training_client_flow.run(options)
