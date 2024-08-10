import argparse
from app.config import config
from app.entity.node_identifier import NodeIdentifier
from app.util import model_utils

options = {
    '-a': ['--aggregation', 'fed_avg', 'name of the aggregation method'],
    '-e': ['--edgebased', False, 'True if edge servers are available otherwise, False'],
    '-dt': ['--decentralized', False, 'True if running in decentralized mode, False otherwise'],
    '-c': ['--clustering', 'none_clustering', 'name of the clustering method'],
    '-s': ['--splitting', 'none_splitting', 'name of the splitting method'],
    '-m': ['--model', 'VGG', 'class name of the training model'],
    '-d': ['--dataset', 'cifar10', 'the name of the using dataset'],
    '-o': ['--offload', False, 'offloading or classic FL mode'],
    '-dl': ['--datasetlink', '', 'the link to dataset  python file'],
    '-ml': ['--modellink', '', 'the link to model  python file'],
    '-i': ['--index', '0', 'the device index'],
    '-ip': ['--ip', 'client1', 'IP address of the node'],
    '-en': ['--energy', 'False', 'enable or disable energy estimation'],
    '-r': ['--rabbitmq-url', 'amqp://rabbitmq:rabbitmq@localhost:5672/%2F', 'RabbitMQ endpoint'],
}


def parse_neighbors(s) -> NodeIdentifier:
    try:
        ip, port = s.split(',')
        ip = ip.strip().strip("'\"")
        port = int(port.strip().strip("'\""))
        return NodeIdentifier(ip, port)
    except:
        raise argparse.ArgumentTypeError("Tuples must be string,int")


def parse_argument(parser: argparse.ArgumentParser):
    """
    Args:
        parser:
    Returns: a map of options with their associated values
    """
    for op in options.keys():
        parser.add_argument(op, options.get(op)[0], help=options.get(op)[2], type=str,
                            default=options.get(op)[1])
    parser.add_argument('-p', '--port', type=int, help='Port number of the node', default=8080)
    parser.add_argument('--neighbors', type=parse_neighbors, nargs='+',
                        help='A list of neighbors in the format "string,int"', default=[])
    args = parser.parse_args()
    option = vars(args)
    option["offload"] = option.get("offload", "False") == "True"
    option["edgebased"] = option.get("edgebased", "False") == "True"
    option["decentralized"] = option.get("decentralized", "False") == 'True'
    if option["decentralized"] and option["edgebased"]:
        raise argparse.ArgumentTypeError("Decentralized and edgebased cannot be both True")

    neighbors = option.get("neighbors")
    if neighbors:
        config.CURRENT_NODE_NEIGHBORS = neighbors

    config.current_node_mq_url = option["rabbitmq_url"]
    config.dataset_name = option.get('dataset')
    config.model_name = option.get('model')
    config.index = int(option.get('index'))
    model_utils.download_model(option.get('modellink'))
    return option
