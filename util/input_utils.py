import argparse

options = {
    '-a': ['--aggregation', 'fed-avg', 'help description'],
    '-c': ['--clustering', 'none', 'help description'],
    '-s': ['--splitting', 'none', 'help description'],
    '-m': ['--model', 'VGG5', 'help description'],
    '-d': ['--dataset', 'cifar-10-python', 'help description'],
    '-o': ['--offload', True, 'FedAdapt or classic FL mode']
}

# returns a map of options with their associated values
def parse_argument(parser: argparse.ArgumentParser()):
    for op in options.keys():
        parser.add_argument(op, options.get(op)[0], help=options.get(op)[2], type=str,
                            default=options.get(op)[1])
    args = parser.parse_args()
    return vars(args)
