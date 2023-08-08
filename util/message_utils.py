"""
websocket messages
"""
initial_global_weights_server_to_client = 'MSG_INITIAL_GLOBAL_WEIGHTS_SERVER_TO_CLIENT'
initial_global_weights_server_to_edge = 'MSG_INITIAL_GLOBAL_WEIGHTS_SERVER_TO_EDGE'
initial_global_weights_edge_to_client = 'MSG_INITIAL_GLOBAL_WEIGHTS_EDGE_TO_CLIENT'

training_time_per_iteration_client_to_server = 'MSG_TRAINING_TIME_PER_ITERATION'

local_weights_client_to_server = 'MSG_LOCAL_WEIGHTS_CLIENT_TO_SERVER'
local_weights_client_to_edge = 'MSG_LOCAL_WEIGHTS_CLIENT_TO_EDGE'
local_weights_edge_to_server = 'MSG_LOCAL_WEIGHTS_EDGE_TO_SERVER'

local_activations_client_to_server = 'MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER'
local_activations_client_to_edge = 'MSG_LOCAL_ACTIVATIONS_CLIENT_TO_EDGE'
local_activations_edge_to_server = 'MSG_LOCAL_ACTIVATIONS_EDGE_TO_SERVER'

split_layers_server_to_edge = 'SPLIT_LAYERS_SERVER_TO_EDGE'
split_layers_edge_to_client = 'SPLIT_LAYERS_EDGE_TO_CLIENT'

test_network = 'MSG_TEST_NETWORK'
client_network = 'MSG_CLIENT_NETWORK'

server_gradients_server_to_client = 'MSG_SERVER_GRADIENTS_SERVER_TO_CLIENT_'
server_gradients_server_to_edge = 'MSG_SERVER_GRADIENTS_SERVER_TO_EDGE_'
server_gradients_edge_to_client = 'MSG_SERVER_GRADIENTS_EDGE_TO_CLIENT_'
