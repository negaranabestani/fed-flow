import argparse
from config import config

options = {
    '-a': ['--aggregation', 'fed_avg', 'help description'],
    '-c': ['--clustering', 'none_clustering', 'help description'],
    '-s': ['--splitting', 'none_splitting', 'help description'],
    '-m': ['--model', 'VGG', 'help description'],
    '-d': ['--dataset', 'cifar10', 'the name of the using dataset'],
    '-o': ['--offload', False, 'FedAdapt or classic FL mode'],
    '-dl': ['--dataset-link', '', 'the link to dataset  python file'],
    '-ml': ['--model-link', '', 'the link to model  python file']
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
    return option

