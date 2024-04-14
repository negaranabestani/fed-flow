import argparse
import sys

sys.path.append('../../../../')
from app.util import input_utils
from app.rl_training.pre_train.flow import preTrain_edge_flow

parser = argparse.ArgumentParser()
options = input_utils.parse_argument(parser)
preTrain_edge_flow.run(options)