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
        rewardTuningParam = preTrain(server, options)

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
                                            rewardTuningParam=rewardTuningParam)
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

    rl_utils.draw_hist(title='Actions',
                       x=actionList,
                       xlabel="Actions",
                       savePath='/Graphs',
                       pictureName='Action_hist_3')

    agent.close()
    msg = [message_utils.finish, True]
    server.scatter(msg)


def preTrain(server, options):
    rewardTuningParams = [0, 0, 0, 0]
    min_Energy = 1.0e20
    max_Energy = 0

    min_trainingTime = 1.0e20
    max_trainingTime = 0

    splittingLayer = rl_utils.allPossibleSplitting(modelLen=config.model_len, deviceNumber=config.K)
    fed_logger.info('====================================>')
    fed_logger.info(f'==> Pre Training Started')

    for splitting in splittingLayer:
        splittingArray = list()
        for i in range(0, len(splitting) - 1, 2):
            op1 = int(splitting[i])
            op2 = int(splitting[i + 1])
            if op1 == 0 or op1 == 2:
                op1 += 1
                if op2 < config.model_len - 1:
                    op2 += 1

            splittingArray.append([op1, op2])

        server.split_layers = splittingArray
        config.split_layer = splittingArray
        fed_logger.info('====================================>')
        fed_logger.info(f'==> Action : {splittingArray}')

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

        fed_logger.info("initializing server")
        server.initialize(server.split_layers, 0.1)

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

        if energy < min_Energy:
            min_Energy = energy
            rewardTuningParams[0] = min_Energy
            min_energy_splitting = splittingArray
            min_Energy_TrainingTime = training_time
        if energy > max_Energy:
            max_Energy = energy
            rewardTuningParams[1] = max_Energy
            max_Energy_splitting = splittingArray
            max_Energy_TrainingTime = training_time

        if training_time < min_trainingTime:
            min_trainingTime = training_time
            rewardTuningParams[2] = min_trainingTime
            min_trainingtime_splitting = splittingArray
            min_trainingTime_energy = energy
        if training_time > max_trainingTime:
            max_trainingTime = training_time
            rewardTuningParams[3] = max_trainingTime
            max_trainingtime_splitting = splittingArray
            max_trainingTime_energy = energy

    fed_logger.info("==> Pre Training Ends")
    fed_logger.info(f"==> Min Energy : {min_Energy}")
    fed_logger.info(f"==> Max Energy : {max_Energy}")
    fed_logger.info(f"==> Min Training Time : {min_trainingTime}")
    fed_logger.info(f"==> Max Training Time : {max_trainingTime}")

    return rewardTuningParams
