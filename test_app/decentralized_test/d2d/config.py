import os
import sys

from app.entity.node_coordinate import NodeCoordinate
from app.entity.node_identifier import NodeIdentifier

DEBUG = os.getenv('DEBUG', 'False') == 'True'
CLIENTS_BANDWIDTH = []
index = 0
simnet = False
# Dataset configration
dataset_name = ''
home = sys.path[0].split('fed-flow')[0] + 'fed-flow' + "/app"
dataset_path = home + '/dataset/data/'
N = 100  # data # length

mq_url = "amqp://rabbitmq:rabbitmq@localhost:5672/"
current_node_mq_url = "Will be set by input options"
mq_host = "localhost"
mq_user = "rabbitmq"
mq_pass = "rabbitmq"
mq_port = "5672"
cluster = "fed-flow"
# Model configration
model_cfg = {
    # (Type, in_channels, out_channels, kernel_size, out_size(c_out*h*w), flops(c_out*h*w*k*k*c_in))
    'VGG5': [('C', 3, 32, 3, 32 * 32 * 32, 32 * 32 * 32 * 3 * 3 * 3), ('M', 32, 32, 2, 32 * 16 * 16, 0),
             ('C', 32, 64, 3, 64 * 16 * 16, 64 * 16 * 16 * 3 * 3 * 32), ('M', 64, 64, 2, 64 * 8 * 8, 0),
             ('C', 64, 64, 3, 64 * 8 * 8, 64 * 8 * 8 * 3 * 3 * 64),
             ('D', 8 * 8 * 64, 128, 1, 64, 128 * 8 * 8 * 64),
             ('D', 128, 10, 1, 10, 128 * 10)]
}
model_name = ''
model_size = 1.28
model_flops = 32.902
total_flops = 8488192
# split_layer = [6]  # Initial split layers for no edge base
split_layer = [[6, 6]]  # Initial split layers
model_len = 7

# FL training configration
R = 2  # FL rounds
learning_rate = 0.01  # Learning rate
B = 100  # Batch size
lr_step_size = 20
lr_gamma = 0.1

# RL training configration
max_episodes = 100  # max training episodes
max_timesteps = 50  # max timesteps in one episode
exploration_times = 20  # exploration times without std decay
n_latent_var = 64  # number of variables in hidden layer
action_std = 0.5  # constant std for action distribution (Multivariate Normal)
update_timestep = 10  # update policy every n timesteps
K_epochs = 50  # update policy for K epochs
eps_clip = 0.2  # clip parameter for PPO
rl_gamma = 0.9  # discount factor
rl_b = 100  # Batchsize
rl_lr = 0.0003  # parameters for Adam optimizer
rl_betas = (0.9, 0.999)
iteration = {'127.0.0.1': 5}  # infer times for each device

random = True
random_seed = 0
# Network configration
SERVER_ADDR = 'server'

SERVER_PORT = 5002
EDGESERVER_PORT = {'edge1': 5001}

K = 6  # Number of devices
G = 1  # Number of groups
S = 1

# Unique clients order
HOST2IP = {}

CLIENTS_LIST = ['client1']
CLIENTS_NAME_TO_INDEX = {'client1': 0}
CLIENTS_INDEX_TO_NAME = {0: 'client1'}
CLIENT_NAME_TO_EDGE_NAME = {'client1': 'edge1'}
CLIENT_NAME_TO_SERVER_NAME = {'client1': 'server1'}

EDGE_SERVER_LIST = ['edge1']
EDGE_SERVER_INDEX_TO_NAME = {0: 'edge1'}
EDGE_NAME_TO_CLIENTS_NAME = {'edge1': ['client1']}

SERVER_INDEX_TO_NAME = {0: 'server1'}

# Topology configration for decentralized mode
CURRENT_NODE_NEIGHBORS: list[NodeIdentifier] = []  # (ip, port)
INITIAL_NODE_COORDINATE = NodeCoordinate