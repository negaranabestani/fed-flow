from abc import ABC

from app.config import config
from app.config.logger import fed_logger
from app.fl_method import fl_method_parser
from app.util import model_utils


class Aggregator(ABC):
    def __init__(self, uninet, split_layers, nets, edge_based, offload):
        self.uninet = uninet
        self.split_layers = split_layers
        self.nets = nets
        self.edge_based = edge_based
        self.offload = offload

    def _aggregate_weights(self, client_ips, aggregate_method, received_weights):
        weight_list = []

        for i, weights in enumerate(received_weights):
            split_point = self.split_layers[i][0] if self.edge_based else self.split_layers[i]

            if split_point != (config.model_len - 1):
                local_weights = (
                    model_utils.concat_weights(self.uninet.state_dict(), weights,
                                               self.nets[client_ips[i]].state_dict()),
                    config.N / config.K
                )
            else:
                local_weights = (weights, config.N / config.K)
            weight_list.append(local_weights)

        zero_model = model_utils.zero_init(self.uninet).state_dict()
        aggregated_model = aggregate_method(zero_model, weight_list, config.N)
        self.uninet.load_state_dict(aggregated_model)

        return aggregated_model

    def aggregate(self, options: dict, received_weights):
        method_name = options.get('aggregation')
        aggregate_method = fl_method_parser.fl_methods.get(method_name)

        if aggregate_method is None:
            fed_logger.error(f"Aggregation method '{method_name}' is not found.")
            return

        self._aggregate_weights(config.CLIENTS_LIST, aggregate_method, received_weights)

    @staticmethod
    def fed_avg(zero_model, weight_list, total_data_size):
        keys = weight_list[0][0].keys()

        for key in keys:
            for weights, beta in weight_list:
                beta_weight = beta / float(total_data_size)
                if 'num_batches_tracked' in key:
                    zero_model[key] = weights[key]
                else:
                    zero_model[key] += weights[key] * beta_weight

        return zero_model
