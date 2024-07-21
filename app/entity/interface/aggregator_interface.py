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

    def aggregate(self, client_ips, aggregate_method, eweights):
        w_local_list = []
        for i in range(len(eweights)):
            sp = self.split_layers[i]
            if self.edge_based:
                sp = self.split_layers[i][0]
            if sp != (config.model_len - 1):
                w_local = (
                    model_utils.concat_weights(self.uninet.state_dict(), eweights[i],
                                               self.nets[client_ips[i]].state_dict()),
                    config.N / config.K)
                w_local_list.append(w_local)
            else:
                w_local = (eweights[i], config.N / config.K)
            w_local_list.append(w_local)

        zero_model = model_utils.zero_init(self.uninet).state_dict()
        aggregated_model = aggregate_method(zero_model, w_local_list, config.N)
        self.uninet.load_state_dict(aggregated_model)
        return aggregated_model

    def call_aggregation(self, options: dict, eweights):
        method = fl_method_parser.fl_methods.get(options.get('aggregation'))
        if method is None:
            fed_logger.error("aggregate method is none")
            return
        self.aggregate(config.CLIENTS_LIST, method, eweights)

    @staticmethod
    def fed_avg(zero_model, w_local_list, total_data_size):
        keys = w_local_list[0][0].keys()

        for k in keys:
            for w in w_local_list:
                beta = float(w[1]) / float(total_data_size)
                if 'num_batches_tracked' in k:
                    zero_model[k] = w[0][k]
                else:
                    zero_model[k] += (w[0][k] * beta)

        return zero_model
