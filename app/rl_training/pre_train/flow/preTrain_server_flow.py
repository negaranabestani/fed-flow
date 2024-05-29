import sys
import time

sys.path.append('../../../../')
from app.config import config
from app.util import model_utils
from app.entity.server import FedServer
from app.config.logger import fed_logger
import numpy as np


def run(options):
    LR = config.LR
    fed_logger.info('Preparing Sever.')
    fed_logger.info('Pre Train Flow Start (Energy of each layer)')
    fed_logger.info('===========================================')

    offload = options.get('offload')
    edge_based = options.get('edgebased')
    if edge_based:
        server = FedServer(options.get('model'), options.get('dataset'), offload, edge_based)
        server.initialize(config.split_layer, LR)
        training_time = 0
        energy = 0

        energyOfLayers = np.zeros((config.K, model_utils.get_unit_model_len()))
        ttOfLayers = np.zeros((config.K, model_utils.get_unit_model_len()))
        fed_logger.info(energyOfLayers)
        x = []

        for layer in range(model_utils.get_unit_model_len()):
            fed_logger.info('====================================>')
            fed_logger.info(f'==> Energy Calculation of Layer {layer} Started.')
            for i in range(5):
                actions = [[layer, layer]] * config.K
                fed_logger.info("=======================================================")
                fed_logger.info(f"Try {i + 1} for Layer {layer}")
                fed_logger.info(f"Actions: {actions}")
                server.split_layers = actions
                s_time = time.time()
                energy_tt_list = rl_flow(server, options, LR)
                for j in range(config.K):
                    energyOfLayers[j][layer] += energy_tt_list[j][0]
                    ttOfLayers[j][layer] += energy_tt_list[j][1]
                e_time = time.time()
                training_time = e_time - s_time

                fed_logger.info("clustering")
                server.cluster(options)
            for j in range(config.K):
                energyOfLayers[j][layer] /= 5
                ttOfLayers[j][layer] /= 5

        fed_logger.info("=======================================================")
        fed_logger.info("Pre Train Ended.")
        fed_logger.info(f"Energy consumption of layer 0 to {model_utils.get_unit_model_len()} of each device is:")
        for i in range(config.K):
            fed_logger.info(f"Device {i}: {energyOfLayers[i]}")
        fed_logger.info("-------------------------------------------------------")
        fed_logger.info(f"Training Time of layer 0 to {model_utils.get_unit_model_len()} of each device is:")
        for i in range(config.K):
            fed_logger.info(f"Device {i}: {ttOfLayers[i]}")

    # x.append(layer)
    # rl_utils.draw_graph(title="Reward vs Episode",
    #                     xlabel="Episode",
    #                     ylabel="Reward",
    #                     figSizeX=10,
    #                     figSizeY=5,
    #                     x=x,
    #                     y=episode_reward,
    #                     savePath='/Graphs',
    #                     pictureName=f"Reward_episode_3")
    #
    # rl_utils.draw_graph(title="Avg Energy vs Episode",
    #                     xlabel="Episode",
    #                     ylabel="Average Energy",
    #                     figSizeX=10,
    #                     figSizeY=5,
    #                     x=x,
    #                     y=episode_energy,
    #                     savePath='/Graphs',
    #                     pictureName=f"Energy_episode_3")
    #
    # rl_utils.draw_graph(title="Avg TrainingTime vs Episode",
    #                     xlabel="Episode",
    #                     ylabel="TrainingTime",
    #                     figSizeX=10,
    #                     figSizeY=5,
    #                     x=x,
    #                     y=episode_trainingTime,
    #                     savePath='/Graphs',
    #                     pictureName=f"TrainingTime_episode_3")
    # fed_logger.info('===========================================')
    # fed_logger.info('Saving Graphs')
    # rl_utils.draw_graph(title="Reward vs Episode",
    #                     xlabel="Episode",
    #                     ylabel="Reward",
    #                     figSizeX=10,
    #                     figSizeY=5,
    #                     x=x,
    #                     y=episode_reward,
    #                     savePath='/Graphs',
    #                     pictureName=f"Reward_episode_3")
    #
    # rl_utils.draw_graph(title="Avg Energy vs Episode",
    #                     xlabel="Episode",
    #                     ylabel="Average Energy",
    #                     figSizeX=10,
    #                     figSizeY=5,
    #                     x=x,
    #                     y=episode_energy,
    #                     savePath='/Graphs',
    #                     pictureName=f"Energy_episode_3")
    #
    # rl_utils.draw_graph(title="Avg TrainingTime vs Episode",
    #                     xlabel="Episode",
    #                     ylabel="TrainingTime",
    #                     figSizeX=10,
    #                     figSizeY=5,
    #                     x=x,
    #                     y=episode_trainingTime,
    #                     savePath='/Graphs',
    #                     pictureName=f"TrainingTime_episode_3")
    #
    # rl_utils.draw_scatter(title="Energy vs TrainingTime",
    #                       xlabel="Energy",
    #                       ylabel="TrainingTime",
    #                       x=episode_energy,
    #                       y=episode_trainingTime,
    #                       savePath='/Graphs',
    #                       pictureName=f"Scatter_3")
    # msg = [message_utils.finish, True]
    # server.scatter(msg)


def rl_flow(server, options, LR):
    fed_logger.info("sending global weights")
    server.edge_offloading_global_weights()
    # fed_logger.info("receiving client network info")
    # server.client_network(config.EDGE_SERVER_LIST)
    #
    # fed_logger.info("test edge servers network")
    # server.test_network(config.EDGE_SERVER_LIST)

    fed_logger.info("preparing state...")
    server.offloading = server.get_offloading(server.split_layers)

    fed_logger.info("clustering")
    server.cluster(options)

    fed_logger.info("getting state")
    offloading = server.split_layers

    # fed_logger.info("splitting")
    # server.split(state, options)
    server.get_split_layers_config_from_edge()

    fed_logger.info("initializing server")
    server.initialize(server.split_layers, LR)

    # fed_logger.info('==> Reinitialization Finish')

    fed_logger.info("start training")
    server.edge_offloading_train(config.CLIENTS_LIST)

    fed_logger.info("receiving local weights")
    local_weights = server.e_local_weights(config.CLIENTS_LIST)

    fed_logger.info("aggregating weights")
    server.call_aggregation(options, local_weights)

    energy_tt_list = server.e_energy_tt(config.CLIENTS_LIST)
    return energy_tt_list
