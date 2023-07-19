from enum import Enum


class Message(Enum):
    test_network = 'MSG_TEST_NETWORK',
    initial_global_weights_server_to_client = 'MSG_INITIAL_GLOBAL_WEIGHTS_SERVER_TO_CLIENT',
    training_time_per_iteration = 'MSG_TRAINING_TIME_PER_ITERATION',
    local_activations = 'MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER',
    server_gradients = 'MSG_SERVER_GRADIENTS_SERVER_TO_CLIENT_',
    local_weights = 'MSG_LOCAL_WEIGHTS_CLIENT_TO_SERVER',
    split_layers = 'SPLIT_LAYERS',
