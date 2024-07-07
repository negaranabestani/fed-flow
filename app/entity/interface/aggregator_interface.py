from abc import ABC, abstractmethod

from app.config.logger import fed_logger
from app.fl_method import fl_method_parser
from app.config import config


class Aggregator(ABC):
    @abstractmethod
    def aggregate(self, client_ips, aggregate_method, eweights):
        pass

    def call_aggregation(self, options: dict, eweights):
        method = fl_method_parser.fl_methods.get(options.get('aggregation'))
        if method is None:
            fed_logger.error("aggregate method is none")
            return
        self.aggregate(config.CLIENTS_LIST, method, eweights)
