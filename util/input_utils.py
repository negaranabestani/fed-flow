import argparse

options = {
    '-a': ['--aggregation', 'fed-avg', 'help description'],
    '-c': ['--clustering', 'none', 'help description'],
    '-s': ['--splitting', 'none', 'help description'],
    '-m': ['--model', 'VGG5', 'help description'],
    '-d': ['--dataset', 'CIFAR10', 'help description']
}


# returns a list of options with their associated values
def server_parse_argument(parser: argparse.ArgumentParser()):
    for op in options.keys():
        parser.add_argument(op, options.get(op)[0], help=options.get(op)[2], type=str,
                            default=options.get(op)[1])
    args = parser.parse_args()
    result = []
    args_map = vars(args)
    for pair in args_map.keys():
        result.append(str(args_map.get(pair)) + '_' + str(pair))

    return result
