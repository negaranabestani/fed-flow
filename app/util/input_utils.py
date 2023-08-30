import argparse
from app.config import config
from app.util import model_utils

options = {
    '-a': ['--aggregation', 'fed_avg', 'help description'],
    '-c': ['--clustering', 'none_clustering', 'help description'],
    '-s': ['--splitting', 'none_splitting', 'help description'],
    '-m': ['--model', 'vgg', 'help description'],
    '-d': ['--dataset', 'cifar10', 'the name of the using dataset'],
    '-o': ['--offload', False, 'FedAdapt or classic FL mode'],
    '-dl': ['--datasetlink', '', 'the link to dataset  python file'],
    '-ml': ['--modellink', '', 'the link to model  python file'],
    '-i': ['--index', '0', 'the device index']
}


def parse_argument(parser: argparse.ArgumentParser()):
    """

    Args:
        parser:

    Returns: a map of options with their associated values

    """
    for op in options.keys():
        parser.add_argument(op, options.get(op)[0], help=options.get(op)[2], type=str,
                            default=options.get(op)[1])
    args = parser.parse_args()
    option = vars(args)
    config.dataset_name = option.get('dataset')
    config.model_name = option.get('model')
    config.index = int(option.get('index'))
    model_utils.download_model(option.get('modellink'))
    return option
