from app.config import config
from app.config.logger import fed_logger
from app.fl_method import fl_method_parser
from app.util import model_utils


class Aggregator:
    def __init__(self, uninet):
        self.uninet = uninet

    @staticmethod
    def get_aggregate_method(aggregation_method_name):
        aggregate_method = fl_method_parser.fl_methods.get(aggregation_method_name)
        if aggregate_method is None:
            fed_logger.error(f"Aggregation method '{aggregation_method_name}' is not found.")
        return aggregate_method

    def aggregate(self, aggregation_method_name, local_weights_list):
        aggregate_method = self.get_aggregate_method(aggregation_method_name)

        if aggregate_method is None:
            return

        zero_model = model_utils.zero_init(self.uninet).state_dict()
        aggregated_model = aggregate_method(zero_model, local_weights_list, config.N)
        self.uninet.load_state_dict(aggregated_model)
        return aggregated_model


