import pickle
import sys
import time

sys.path.append('../../../')
from app.config import config
from app.util import model_utils, message_utils, rl_utils
from app.entity.server import FedServer
from app.config.logger import fed_logger
import numpy as np


def run(options):
    LR = config.LR
    fed_logger.info('Preparing Sever.')
    fed_logger.info("start mode: " + str(options.values()))
    offload = options.get('offload')
    edge_based = options.get('edgebased')
    if edge_based:

        server = FedServer(config.SERVER_ADDR, config.SERVER_PORT, options.get('model'),
                           options.get('dataset'), offload, edge_based)

        agent = rl_utils.createAgent(agentType='tensorforce', fraction=0.8, timestepNum=config.max_timesteps,
                                     saveSummariesPath=None)
        server.initialize(config.split_layer, LR)
        training_time = 0
        energy = 0

        res = {}
        res['training_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []
        episode_energy, episode_trainingTime, episode_reward = [], [], []
        timestep_energy, timestep_trainingTime, timestep_reward = [], [], []
        x = []
        actionList = []

        classicFlTrainingTime, maxEnergy = preTrain(server, options)

        for r in range(config.max_episodes):
            fed_logger.info('====================================>')
            fed_logger.info(f'==> Episode {r} Start')

            s_time = time.time()

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

            fed_logger.info("splitting")
            randActions = np.random.uniform(low=0.0, high=1.0, size=(config.K * 2))
            actions = []
            for i in range(0, len(randActions), 2):
                actions.append([rl_utils.actionToLayerEdgeBase([randActions[i], randActions[i + 1]])[0],
                                rl_utils.actionToLayerEdgeBase([randActions[i], randActions[i + 1]])[1]])

            server.split_layers = actions
            server.split_layer()

            fed_logger.info("initializing server")
            server.initialize(server.split_layers, LR)

            # fed_logger.info('==> Reinitialization Finish')

            fed_logger.info("start training")
            server.edge_offloading_train(config.CLIENTS_LIST)

            fed_logger.info("receiving local weights")
            local_weights = server.e_local_weights(config.CLIENTS_LIST)

            fed_logger.info("aggregating weights")
            server.call_aggregation(options, local_weights)

            energy = server.e_energy(config.CLIENTS_LIST)
            e_time = time.time()
            # Recording each round training time, bandwidth and test_app accuracy
            training_time = e_time - s_time
            state = server.edge_based_state(training_time, offloading, energy)
            fed_logger.info("state: " + str(state))

            # res['training_time'].append(training_time)
            # res['bandwidth_record'].append(server.bandwith())
            # with open(config.home + '/results/FedAdapt_res.pkl', 'wb') as f:
            #     pickle.dump(res, f)
            #
            # fed_logger.info("testing accuracy")
            # test_acc = model_utils.test(server.uninet, server.testloader, server.device, server.criterion)
            # res['test_acc_record'].append(test_acc)

            fed_logger.info('Round Finish')
            fed_logger.info('==> Round Training Time: {:}'.format(training_time))

            for timestep in range(config.max_timesteps):
                floatAction = agent.act(states=state, evaluation=False)
                actions = []
                for i in range(0, len(floatAction), 2):
                    actions.append([rl_utils.actionToLayerEdgeBase([floatAction[i], floatAction[i + 1]])[0],
                                    rl_utils.actionToLayerEdgeBase([floatAction[i], floatAction[i + 1]])[1]])

                actionList.append(actions)
                server.split_layers = actions

                fed_logger.info('====================================>')
                fed_logger.info(f'==> Timestep {timestep} Start')

                s_time = time.time()

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
                server.split_layer()

                if r > 49:
                    LR = config.LR * 0.1

                fed_logger.info("initializing server")
                server.initialize(server.split_layers, LR)

                # fed_logger.info('==> Reinitialization Finish')

                fed_logger.info("start training")
                server.edge_offloading_train(config.CLIENTS_LIST)

                fed_logger.info("receiving local weights")
                local_weights = server.e_local_weights(config.CLIENTS_LIST)

                fed_logger.info("aggregating weights")
                server.call_aggregation(options, local_weights)

                energy = server.e_energy(config.CLIENTS_LIST)
                e_time = time.time()

                # Recording each round training time, bandwidth and test_app accuracy
                training_time = e_time - s_time

                fed_logger.info('Updating Agent')
                reward = rl_utils.rewardFun(fraction=0.8, energy=energy, trainingTime=training_time,
                                            classicFlTrainingTime=classicFlTrainingTime, maxEnergy=maxEnergy)
                agent.observe(terminal=False, reward=reward)

                timestep_energy.append(energy)
                timestep_trainingTime.append(training_time)
                timestep_reward.append(reward)

                state = server.edge_based_state(training_time, offloading, energy)
                fed_logger.info("state: " + str(state))

                # res['training_time'].append(training_time)
                # res['bandwidth_record'].append(server.bandwith())
                # with open(config.home + '/results/FedAdapt_res.pkl', 'wb') as f:
                #     pickle.dump(res, f)
                #
                # fed_logger.info("testing accuracy")
                # test_acc = model_utils.test(server.uninet, server.testloader, server.device, server.criterion)
                # res['test_acc_record'].append(test_acc)

                fed_logger.info('Round Finish')
                fed_logger.info('==> Round Training Time: {:}'.format(training_time))

            x.append(r)
            episode_energy.append(sum(timestep_energy) / config.max_timesteps)
            episode_trainingTime.append(sum(timestep_trainingTime) / config.max_timesteps)
            episode_reward.append(sum(timestep_reward) / config.max_timesteps)
            timestep_energy, timestep_trainingTime, timestep_reward = [], [], []

    fed_logger.info('===========================================')
    fed_logger.info('Saving Graphs')
    rl_utils.draw_graph(title="Reward vs Episode",
                        xlabel="Episode",
                        ylabel="Reward",
                        figSizeX=10,
                        figSizeY=5,
                        x=x,
                        y=episode_reward,
                        savePath='/Graphs',
                        pictureName=f"Reward_episode_3")

    rl_utils.draw_graph(title="Avg Energy vs Episode",
                        xlabel="Episode",
                        ylabel="Average Energy",
                        figSizeX=10,
                        figSizeY=5,
                        x=x,
                        y=episode_energy,
                        savePath='/Graphs',
                        pictureName=f"Energy_episode_3")

    rl_utils.draw_graph(title="Avg TrainingTime vs Episode",
                        xlabel="Episode",
                        ylabel="TrainingTime",
                        figSizeX=10,
                        figSizeY=5,
                        x=x,
                        y=episode_trainingTime,
                        savePath='/Graphs',
                        pictureName=f"TrainingTime_episode_3")

    rl_utils.draw_scatter(title="Energy vs TrainingTime",
                          xlabel="Energy",
                          ylabel="TrainingTime",
                          x=episode_energy,
                          y=episode_trainingTime,
                          savePath='/Graphs',
                          pictureName=f"Scatter_3")

    # rl_utils.draw_hist(title='Actions',
    #                    x=actionList,
    #                    xlabel="Actions",
    #                    savePath='/Graphs',
    #                    pictureName='Action_hist_3')

    agent.close()
    msg = [message_utils.finish, True]
    server.scatter(msg)


# this method return maxEnergy and classicFL training Time for reward tuning
def preTrain(server, options) -> tuple[float, float]:
    classicFLTrainingTime = 0
    maxEnergySplitting = []
    maxEnergy = 0
    energyArray = []
    trainingTimeArray = []

    splittingLayer = rl_utils.allPossibleSplitting(modelLen=config.model_len, deviceNumber=1)
    fed_logger.info('====================================>')
    fed_logger.info(f'==> Pre Training Started')

    for splitting in splittingLayer:
        splittingArray = list()
        for i in range(0, len(splitting) - 1, 2):
            op1 = int(splitting[i])
            op2 = int(splitting[i + 1])
            splittingArray.append([op1, op2])

        server.split_layers = splittingArray

        fed_logger.info('====================================>')
        fed_logger.info(f'==> Action : {splittingArray}')

        s_time = time.time()

        # fed_logger.info("sending global weights")
        server.edge_offloading_global_weights()

        # fed_logger.info("receiving client network info")
        # server.client_network(config.EDGE_SERVER_LIST)

        # fed_logger.info("test edge servers network")
        # server.test_network(config.EDGE_SERVER_LIST)

        # fed_logger.info("preparing state...")
        server.offloading = server.get_offloading(server.split_layers)

        # fed_logger.info("clustering")
        server.cluster(options)

        # fed_logger.info("getting state")
        offloading = server.split_layers

        # fed_logger.info("splitting")
        # server.split(state, options)
        server.split_layer()

        # fed_logger.info("initializing server")
        server.initialize(server.split_layers, 0.1)

        # fed_logger.info('==> Reinitialization Finish')

        # fed_logger.info("start training")
        server.edge_offloading_train(config.CLIENTS_LIST)

        # fed_logger.info("receiving local weights")
        local_weights = server.e_local_weights(config.CLIENTS_LIST)

        # fed_logger.info("aggregating weights")
        server.call_aggregation(options, local_weights)

        energy = server.e_energy(config.CLIENTS_LIST)
        e_time = time.time()

        # Recording each round training time, bandwidth and test_app accuracy
        training_time = e_time - s_time

        state = server.edge_based_state(training_time, offloading, energy)
        # fed_logger.info("state: " + str(state))
        fed_logger.info(f"Energy of Action {splittingArray} : {energy}")
        fed_logger.info(f"Training Time of Action {splittingArray} : {training_time}")

        if splittingArray == [[config.model_len - 1, config.model_len - 1]]:
            fed_logger.info("====================================>")
            fed_logger.info("Classic FL Energy ")
            fed_logger.info(f"Energy : {energy}")
            fed_logger.info("Classic FL Training Time ")
            fed_logger.info(f"TrainingTime : {training_time}")
            classicFLTrainingTime = training_time

        energyArray.append(energy)
        trainingTimeArray.append(training_time)

        if energy > maxEnergy:
            maxEnergy = energy
            maxEnergySplitting = splittingArray[0]

    server.split_layers = []
    for _ in range(config.K):
        server.split_layers.append(maxEnergySplitting)

    # fed_logger.info("sending global weights")
    server.edge_offloading_global_weights()

    # fed_logger.info("receiving client network info")
    # server.client_network(config.EDGE_SERVER_LIST)

    # fed_logger.info("test edge servers network")
    # server.test_network(config.EDGE_SERVER_LIST)

    # fed_logger.info("preparing state...")
    server.offloading = server.get_offloading(server.split_layers)

    # fed_logger.info("clustering")
    server.cluster(options)

    # fed_logger.info("getting state")
    offloading = server.split_layers

    # fed_logger.info("splitting")
    # server.split(state, options)
    server.split_layer()

    # fed_logger.info("initializing server")
    server.initialize(server.split_layers, 0.1)

    # fed_logger.info('==> Reinitialization Finish')

    # fed_logger.info("start training")
    server.edge_offloading_train(config.CLIENTS_LIST)

    # fed_logger.info("receiving local weights")
    local_weights = server.e_local_weights(config.CLIENTS_LIST)

    # fed_logger.info("aggregating weights")
    server.call_aggregation(options, local_weights)

    energy = server.e_energy(config.CLIENTS_LIST)
    maxEnergy = energy

    rl_utils.draw_hist(title='Energy',
                       x=energyArray,
                       xlabel="Energy",
                       savePath='/Graphs',
                       pictureName='energy_hist_3')
    rl_utils.draw_hist(title='trainingTime',
                       x=trainingTimeArray,
                       xlabel="TrainingTime",
                       savePath='/Graphs',
                       pictureName='trainingTime_hist_3')

    fed_logger.info("====================================>")
    fed_logger.info("==> Pre Training Ends")
    fed_logger.info(f"==> Max Energy : {maxEnergy}")
    fed_logger.info(f"==> Max Energy Splitting : {maxEnergySplitting}")
    fed_logger.info(f"==> Classic FL Training Timme : {classicFLTrainingTime}")

    return classicFLTrainingTime, maxEnergy
