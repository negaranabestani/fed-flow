from abc import ABC

from app.config import config
from app.config.logger import fed_logger
from app.fl_method import fl_method_parser
from app.util import model_utils


class Aggregator(ABC):
    def __init__(self, uninet):
        self.uninet = uninet

    def aggregate(self, aggregation_method_name, local_weights_list):
        aggregate_method = fl_method_parser.fl_methods.get(aggregation_method_name)

        if aggregate_method is None:
            fed_logger.error(f"Aggregation method '{aggregation_method_name}' is not found.")
            return

        zero_model = model_utils.zero_init(self.uninet).state_dict()
        aggregated_model = aggregation_method_name(zero_model, local_weights_list, config.N)
        self.uninet.load_state_dict(aggregated_model)
        return aggregated_model
