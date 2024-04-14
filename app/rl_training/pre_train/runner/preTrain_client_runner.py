import argparse
import sys

sys.path.append('../../../../')
from app.util import input_utils
from app.rl_training.pre_train.flow import preTrain_client_flow

parser = argparse.ArgumentParser()
options = input_utils.parse_argument(parser)
preTrain_client_flow.run(options)
