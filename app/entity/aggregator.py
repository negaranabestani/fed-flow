from abc import ABC

from app.config import config
from app.config.logger import fed_logger
from app.entity.aggregator_config import AggregatorConfig
from app.fl_method import fl_method_parser
from app.util import model_utils


class Aggregator(ABC):
    def __init__(self, config: AggregatorConfig):
        self.uninet = config.uninet
        self.split_layers = config.split_layers
        self.nets = config.nets
        self.edge_based = config.edge_based

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
        aggregate_method = fl_method_parser.fl_methods.get(options.get('aggregation'))

        if aggregate_method is None:
            fed_logger.error(f"Aggregation method '{method_name}' is not found.")
            return

        self._aggregate_weights(config.CLIENTS_LIST, aggregate_method, received_weights)
