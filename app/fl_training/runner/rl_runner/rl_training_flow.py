import pickle
import socket
import sys
import time

sys.path.append('../../../../')
from app.config import config
from app.util import model_utils, message_utils, rl_utils
from app.fl_training.entity.fed_server import FedServer
from app.config.logger import fed_logger


def run(options):
    LR = config.LR
    fed_logger.info('Preparing Sever.')
    fed_logger.info("start mode: " + str(options.values()))
    offload = options.get('offload')
    edge_based = options.get('edgebased')
    if edge_based:

        server = FedServer(config.SERVER_ADDR, config.SERVER_PORT, options.get('model'),
                           options.get('dataset'), offload, edge_based)

        env = rl_utils.createEnv(timestepNum=config.max_timesteps, fraction=0.8, edgeDeviceNum=config.S,
                                 iotDeviceNum=config.K, rewardTuningParams=[0, 0, 0, 0])

        agent = rl_utils.createAgent(agentType='tensorforce', fraction=0.8, environment=env,
                                     timestepNum=config.max_timesteps, saveSummariesPath=None)
        server.initialize(config.split_layer, LR)
        training_time = 0
        energy = 0
        res = {}
        res['training_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []

        for r in range(config.max_episodes):
            episode_energy = list()
            episode_trainingTime = list()
            episode_reward = list()

            states = env.reset()
            internals = agent.initial_internals()
            terminal = False
            for timestep in range(config.max_timesteps):

                fed_logger.info('====================================>')
                fed_logger.info('==> Round {:} Start'.format(r))

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

                state = server.edge_based_state(training_time, offloading, energy)
                fed_logger.info("state: " + str(state))

                fed_logger.info("splitting")
                server.split(state, options)
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

                res['training_time'].append(training_time)
                res['bandwidth_record'].append(server.bandwith())
                with open(config.home + '/results/FedAdapt_res.pkl', 'wb') as f:
                    pickle.dump(res, f)

                fed_logger.info("testing accuracy")
                test_acc = model_utils.test(server.uninet, server.testloader, server.device, server.criterion)
                res['test_acc_record'].append(test_acc)

                fed_logger.info('Round Finish')
                fed_logger.info('==> Round Training Time: {:}'.format(training_time))

    msg = [message_utils.finish, True]
    server.scatter(msg)


floatAction = agent.act(states=state, evaluation=False)
actions = []
for i in range(0, len(floatAction), 2):
    actions.append([rl_utils.actionToLayerEdgeBase([floatAction[i], floatAction[i + 1]])[0],
                    rl_utils.actionToLayerEdgeBase([floatAction[i], floatAction[i + 1]])[1]])
server.split_layers = actions

reward = rl_utils.rewardFun(energy=energy, trainingTime=training_time)
agent.observe(terminal=False, reward=reward)