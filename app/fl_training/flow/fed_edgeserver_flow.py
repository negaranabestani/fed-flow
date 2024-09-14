import sys
import threading
import time

from app.entity.aggregators.factory import create_aggregator
from app.entity.decentralized_edge_server import FedDecentralizedEdgeServer
from app.entity.http_communicator import HTTPCommunicator
from app.entity.node_type import NodeType
from app.util import rl_utils, model_utils

sys.path.append('../../../')
from app.config import config
from app.config.logger import fed_logger
from app.entity.edge_server import FedEdgeServer


def run_offload(server: FedEdgeServer, LR, estimate_energy):
    server.initialize(config.split_layer, LR,
                      config.EDGE_NAME_TO_CLIENTS_NAME[config.EDGE_SERVER_INDEX_TO_NAME[config.index]])

    res = {}
    res['trianing_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []
    client_ips = config.EDGE_NAME_TO_CLIENTS_NAME[config.EDGE_SERVER_INDEX_TO_NAME[config.index]]
    for r in range(config.R):
        config.current_round = r
        fed_logger.info('====================================>')
        fed_logger.info('==> Round {:} Start'.format(r + 1))
        fed_logger.info("receiving global weights")
        server.get_global_weights(client_ips)
        fed_logger.info("test clients network")
        server.test_client_network(client_ips)
        fed_logger.info("sending clients network")
        server.client_network()
        fed_logger.info("test server network")
        server.test_server_network()
        fed_logger.info("receiving and sending splitting info")
        server.get_split_layers_config(client_ips)
        fed_logger.info("initializing server")
        server.initialize(server.split_layers, LR, client_ips)
        threads = {}
        fed_logger.info("start training")
        for i in range(len(client_ips)):
            threads[client_ips[i]] = threading.Thread(target=server.thread_offload_training,
                                                      args=(client_ips[i],), name=client_ips[i])
            threads[client_ips[i]].start()

        for i in range(len(client_ips)):
            threads[client_ips[i]].join()
        if estimate_energy:
            server.energy(client_ips)
        if r > 49:
            LR = config.LR * 0.1

        # energy = float(energy_estimation.energy())
        # energy /= batch_num
        # fed_logger.info(Fore.LIGHTBLUE_EX + f"Energy : {energy}" + Fore.RESET)
        fed_logger.info('==> Round {:} End'.format(r + 1))


def run_no_offload(server: FedEdgeServer, LR):
    server.initialize(config.split_layer, LR,
                      config.EDGE_NAME_TO_CLIENTS_NAME[config.EDGE_SERVER_INDEX_TO_NAME[config.index]])
    res = {}
    res['trianing_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []
    client_ips = config.EDGE_NAME_TO_CLIENTS_NAME[config.EDGE_SERVER_INDEX_TO_NAME[config.index]]
    for r in range(config.R):
        config.current_round = r
        fed_logger.info('====================================>')
        fed_logger.info('==> Round {:} Start'.format(r))
        fed_logger.info("receiving global weights")
        server.no_offload_global_weights()
        # fed_logger.info("test clients network")
        # server.test_client_network(client_ips)
        # fed_logger.info("sending clients network")
        # server.client_network()
        # fed_logger.info("test server network")
        # server.test_server_network()
        threads = {}
        fed_logger.info("start training")
        for i in range(len(client_ips)):
            threads[client_ips[i]] = threading.Thread(target=server.thread_no_offload_training,
                                                      args=(client_ips[i],), name=client_ips[i])
            threads[client_ips[i]].start()

        for i in range(len(client_ips)):
            threads[client_ips[i]].join()
        if r > 49:
            LR = config.LR * 0.1


def run_decentralized(edge_server: FedDecentralizedEdgeServer, learning_rate, options: dict):
    edge_server.initialize(learning_rate)
    client_bw, edge_bw = [], []
    training_times = []
    rounds = []
    accuracy = []
    for r in range(config.R):
        config.current_round = r
        rounds.append(r)
        fed_logger.info('====================================>')
        fed_logger.info('==> Round {:} Start'.format(r + 1))

        fed_logger.info("sending global weights")
        edge_server.scatter_global_weights([NodeType.CLIENT])

        s_time = time.time()

        fed_logger.info("gathering neighbors network speed")
        edge_server.gather_neighbors_network_bandwidth()

        fed_logger.info("clustering")
        edge_server.cluster(options)

        fed_logger.info("getting neighbors bandwidth")
        neighbors_bandwidth = edge_server.get_neighbors_bandwidth()
        neighbors_bandwidth_by_type: dict[NodeType, list[float]] = {}
        for neighbor, bw in neighbors_bandwidth.items():
            neighbor_type = HTTPCommunicator.get_node_type(neighbor)
            if neighbor_type not in neighbors_bandwidth_by_type:
                neighbors_bandwidth_by_type[neighbor_type] = []
            neighbors_bandwidth_by_type[neighbor_type].append(bw.bandwidth)
        client_bw.append(
            sum(neighbors_bandwidth_by_type[NodeType.CLIENT]) / len(neighbors_bandwidth_by_type[NodeType.CLIENT]))
        edge_bw.append(
            sum(neighbors_bandwidth_by_type[NodeType.EDGE]) / len(neighbors_bandwidth_by_type[NodeType.EDGE]))

        fed_logger.info("splitting")
        edge_server.split(neighbors_bandwidth_by_type[NodeType.CLIENT], options)
        fed_logger.info(f"Split Config : {edge_server.split_layers}")
        edge_server.scatter_split_layers()

        fed_logger.info("start training")
        edge_server.start_decentralized_training()

        fed_logger.info("receiving local weights")
        local_weights = edge_server.gather_local_weights()

        fed_logger.info("aggregating weights")
        edge_server.aggregate(local_weights)

        fed_logger.info("start gossiping with neighbors")
        edge_server.gossip_with_neighbors()
        #
        e_time = time.time()

        # Recording each round training time, bandwidth and test_app accuracy
        training_time = e_time - s_time
        training_times.append(training_time)

        fed_logger.info("testing accuracy")
        test_acc = model_utils.test(edge_server.uninet, edge_server.testloader, edge_server.device, edge_server.criterion)
        fed_logger.info(f"Test Accuracy : {test_acc}")
        accuracy.append(test_acc)
        fed_logger.info('Round Finish')
        fed_logger.info('==> Round {:} End'.format(r + 1))
        fed_logger.info('==> Round Training Time: {:}'.format(training_time))

    current_time = time.strftime("%Y-%m-%d %H:%M")
    runtime_config = f'{current_time} offload decentralized'
    rl_utils.draw_graph(10, 5, rounds, training_times, f"Edge {str(edge_server)} Training time", "FL Rounds",
                        "Training Time (s)",
                        f"Graphs/{runtime_config}",
                        f"trainingTime-{str(edge_server)}", True)
    rl_utils.draw_graph(10, 5, rounds, client_bw, f"Edge {str(edge_server)} Average clients BW", "FL Rounds",
                        "clients BW (bytes / s)",
                        f"Graphs/{runtime_config}",
                        f"client_bw-{str(edge_server)}", True)
    rl_utils.draw_graph(10, 5, rounds, edge_bw, f"Edge {str(edge_server)} Average edges BW", "FL Rounds",
                        "edges BW (bytes / s)",
                        f"Graphs/{runtime_config}",
                        f"edge_bw-{str(edge_server)}", True)
    rl_utils.draw_graph(10, 5, rounds, accuracy, f"Edge {str(edge_server)} Accuracy", "FL Rounds", "accuracy",
                        f"Graphs/{runtime_config}",
                        f"accuracy-{str(edge_server)}", True)


def run_centralized(edge_server: FedDecentralizedEdgeServer, learning_rate):
    edge_server.initialize(learning_rate)
    for r in range(config.R):
        config.current_round = r
        fed_logger.info('====================================>')
        fed_logger.info('==> Round {:} Start'.format(r + 1))
        fed_logger.info("receiving and sending global weights")
        edge_server.gather_and_scatter_global_weight()
        fed_logger.info("test clients network")
        edge_server.gather_neighbors_network_bandwidth()
        fed_logger.info("receiving and sending splitting info")
        edge_server.gather_and_scatter_split_config()
        fed_logger.info("start training")
        edge_server.start_centralized_training()
        fed_logger.info('==> Round {:} End'.format(r + 1))


def run(options_ins):
    LR = config.LR
    fed_logger.info('Preparing Sever.')
    offload = options_ins.get('offload')
    decentralized = options_ins.get('decentralized')
    estimate_energy = options_ins.get('energy') == "True"
    if not decentralized:
        edge_server = FedEdgeServer(
            options_ins.get('ip'), options_ins.get('port'), options_ins.get('model'),
            options_ins.get('dataset'), offload=offload)
    else:
        aggregator = create_aggregator(options_ins.get('aggregation'))
        edge_server = FedDecentralizedEdgeServer(options_ins.get('ip'), options_ins.get('port'),
                                                 options_ins.get('model'),
                                                 options_ins.get('dataset'), offload, aggregator)

    if decentralized:
        edge_server.add_neighbors(config.CURRENT_NODE_NEIGHBORS)
        fed_logger.info("neighbors: " + str(config.CURRENT_NODE_NEIGHBORS))

    fed_logger.info("start mode: " + str(options_ins.values()))
    if decentralized:
        run_decentralized(edge_server, LR, options_ins)
    else:
        if offload:
            run_offload(edge_server, LR, estimate_energy)
        else:
            run_no_offload(edge_server, LR)
    edge_server.stop_server()
