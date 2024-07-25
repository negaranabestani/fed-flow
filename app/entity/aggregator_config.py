class AggregatorConfig:
    def __init__(self, aggregation_method, uninet, split_layers, nets, edge_based):
        self.aggregation_method = aggregation_method
        self.uninet = uninet
        self.split_layers = split_layers
        self.nets = nets
        self.edge_based = edge_based
