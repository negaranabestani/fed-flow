import sys
import time

sys.path.append('../../../')
from app.config import config
from app.util import message_utils, rl_utils
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

        server = FedServer(options.get('model'), options.get('dataset'), offload, edge_based)

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

        classicFlTrainingTime, cl_energy = preTrain(server, options)

        for r in range(config.max_episodes):
            fed_logger.info('====================================>')
            fed_logger.info(f'==> Episode {r} Start')

            fed_logger.info("sending global weights")
            server.edge_offloading_global_weights()
            s_time = time.time()

            # fed_logger.info("receiving client network info")
            # server.client_network(config.EDGE_SERVER_LIST)
            #
            # fed_logger.info("test edge servers network")
            # server.test_network(config.EDGE_SERVER_LIST)

            fed_logger.info("clustering")
            server.cluster(options)

            fed_logger.info("splitting")
            randActions = np.random.uniform(low=0.0, high=1.0, size=(config.K * 2))
            actions = []
            for i in range(0, len(randActions), 2):
                actions.append([rl_utils.actionToLayerEdgeBase([randActions[i], randActions[i + 1]])[0],
                                rl_utils.actionToLayerEdgeBase([randActions[i], randActions[i + 1]])[1]])

            server.split_layers = [[3,4]]
            # fed_logger.info("preparing state...")
            server.offloading = server.get_offloading(server.split_layers)

            # fed_logger.info("getting state")
            offloading = server.split_layers

            fed_logger.info(f'ACTION : {actions}')
            server.split_layer()

            fed_logger.info("initializing server")
            server.initialize(server.split_layers, LR)

            # fed_logger.info('==> Reinitialization Finish')

            fed_logger.info("start training")
            server.edge_offloading_train(config.CLIENTS_LIST)

            fed_logger.info("receiving local weights")
            local_weights = server.e_local_weights(config.CLIENTS_LIST)

            fed_logger.info("aggregating weights starts")
            server.call_aggregation(options, local_weights)
            fed_logger.info("aggregating weights ends")

            fed_logger.info("creating energy_tt starts")
            energy_tt_list = server.e_energy_tt(config.CLIENTS_LIST)
            fed_logger.info("creating energy_tt ends.")

            e_time = time.time()
            training_time = e_time - s_time

            # Recording each round training time, bandwidth and test_app accuracy
            state = server.edge_based_state(offloading, energy_tt_list, training_time)
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
                fed_logger.info('====================================>')
                fed_logger.info(f'==> Timestep {timestep} Start')

                floatAction = agent.act(states=state, evaluation=False)
                actions = []
                for i in range(0, len(floatAction), 2):
                    actions.append([rl_utils.actionToLayerEdgeBase([floatAction[i], floatAction[i + 1]])[0],
                                    rl_utils.actionToLayerEdgeBase([floatAction[i], floatAction[i + 1]])[1]])

                fed_logger.info(f'ACTION : {actions}')
                actionList.append(actions)
                server.split_layers = [[1,3]]

                s_time = time.time()
                energy_tt_list = rl_flow(server, options, r, LR)

                e_time = time.time()
                training_time = e_time - s_time

                fed_logger.info("preparing state...")
                server.offloading = server.get_offloading(server.split_layers)

                fed_logger.info("clustering")
                server.cluster(options)

                fed_logger.info("getting state")
                offloading = server.split_layers

                state = server.edge_based_state(offloading, energy_tt_list, training_time)

                # Recording each round training time, bandwidth and test_app accuracy
                fed_logger.info('Updating Agent')
                reward = rl_utils.rewardFunTan(fraction=0.8, energy=state[0], trainingTime=training_time,
                                               classicFlTrainingTime=classicFlTrainingTime, classic_Fl_Energy=cl_energy)
                agent.observe(terminal=False, reward=reward)

                timestep_energy.append(state[0])
                timestep_trainingTime.append(training_time)
                timestep_reward.append(reward)

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


# this method return classicFL training Time and energy for reward tuning
def preTrain(server, options) -> tuple[float, float]:
    classicFLTrainingTime = 0
    energyArray = []
    trainingTimeArray = []
    fed_logger.info('====================================>')
    fed_logger.info(f'==> Pre Training Started')

    for j in range(5):
        fed_logger.info(f"Try {j + 1}/5")
        fed_logger.info('====================================>')
        # for splitting in splittingLayer:
        #     splittingArray = list()
        #     for char in splitting:
        #         splittingArray.append(int(char))
        splittingArray = [config.model_len - 1, config.model_len - 1]
        server.split_layers = [splittingArray * config.K]

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

        energy_tt_list = server.e_energy_tt(config.CLIENTS_LIST)
        e_time = time.time()
        training_time = e_time - s_time

        state = server.edge_based_state(offloading, energy_tt_list, training_time)

        # Recording each round training time, bandwidth and test_app accuracy
        fed_logger.info("====================")
        fed_logger.info(f"Action : {offloading}")
        fed_logger.info(f"state: {state}")
        fed_logger.info(f"Energy of Action {offloading} : {state[0]}")
        fed_logger.info(f"Training Time of Action {offloading} : {state[1]}")
        fed_logger.info("====================================>")
        fed_logger.info("Classic FL Energy ")
        fed_logger.info(f"Energy : {state[0]}")
        fed_logger.info("Classic FL Training Time ")
        fed_logger.info(f"TrainingTime : {state[1]}")

        energyArray.append(state[0])
        trainingTimeArray.append(state[1])

    fed_logger.info("====================================>")
    fed_logger.info(f"Tries Finished.")
    for i in range(5):
        fed_logger.info(f"Try {i + 1}/5 :")
        fed_logger.info(f"==> classic-fl Energy : {energyArray[i]}")
        fed_logger.info(f"==> classic-fl Training Time : {trainingTimeArray[i]}")
    fed_logger.info("====================================>")

    rl_utils.draw_hist(title='Energy',
                       x=energyArray,
                       xlabel="Energy",
                       savePath='/Graphs',
                       pictureName='PreTrain_energy_hist')
    rl_utils.draw_hist(title='trainingTime',
                       x=trainingTimeArray,
                       xlabel="TrainingTime",
                       savePath='/Graphs',
                       pictureName='PreTrain_trainingTime_hist')

    fed_logger.info("====================================>")
    fed_logger.info(f"==> Pre Training Ends")
    # fed_logger.info(f"==> Max Energy Splitting : {maxEnergySplitting}")
    # fed_logger.info(f"==> Max Energy : {maxEnergy}")
    # fed_logger.info(f"==> Min Energy Splitting : {minEnergySplitting}")
    # fed_logger.info(f"==> Min Energy : {minEnergy}")
    fed_logger.info(f"==> Classic FL Energy : {sum(energyArray) / len(energyArray)}")
    fed_logger.info(f"==> Classic FL Training Timme : {sum(trainingTimeArray) / len(trainingTimeArray)}")
    fed_logger.info("====================================>")

    return sum(trainingTimeArray) / len(trainingTimeArray), sum(energyArray) / len(energyArray)


def rl_flow(server, options, r, LR):
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

    energy_tt_list = server.e_energy_tt(config.CLIENTS_LIST)
    return energy_tt_list
