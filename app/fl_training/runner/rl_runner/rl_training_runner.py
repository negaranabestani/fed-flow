import argparse
import sys

sys.path.append('../../../../')
from app.util import input_utils
from app.fl_training.runner.rl_runner import rl_training_flow

parser = argparse.ArgumentParser()
options = input_utils.parse_argument(parser)
rl_training_flow.run(options)
