import sys

CLIENTS_BANDWIDTH = []
index = 0
simnet = False
# Dataset configration
dataset_name = ''
home = sys.path[0].split('fed-flow')[0] + 'fed-flow' + "/app"
dataset_path = home + '/dataset/data/'
N = 100  # data length
# mq_url = "sparrow.rmq.cloudamqp.com"
mq_port = 5672
mq_url = "amqp://user:password@broker:5672/%2F"
mq_host = "edge1"
mq_user = "user"
mq_pass = "password"
mq_vh = "/"
cluster = "fed-flow"
current_round = 0
model_name = ''
model_size = 1.28
model_flops = 32.902
total_flops = 8488192
split_layer = [[6, 6]]  # Initial split layers
model_len = 7

# FL training configration
R = 6  # FL rounds
LR = 0.01  # Learning rate
B = 100  # Batch size

# RL training configration
max_episodes = 2000  # max training episodes
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

K = 1  # Number of devices
G = 1  # Number of groups
S = 1

# Unique clients order
HOST2IP = {}
CLIENTS_INDEX = {0: 'client1'}
CLIENTS_CONFIG = {'client1': 0}
EDGE_SERVER_LIST = ['edge1']
EDGE_SERVER_CONFIG = {0: 'edge1'}
CLIENTS_LIST = ['client1']
EDGE_MAP = {'edge1': ['client1']}
CLIENT_MAP = {'client1': 'edge1'}
