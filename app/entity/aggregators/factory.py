from app.config import config
from app.entity.aggregators.base_aggregator import BaseAggregator
from app.entity.aggregators.fed_avg import FedAvg


def create_aggregator(aggregator_name: str) -> BaseAggregator:
    normalized_aggregator_name = aggregator_name.lower().replace('-', '').replace('_', '')
    if normalized_aggregator_name == 'fedavg':
        return FedAvg(config.N)
    else:
        raise ValueError(f"Aggregator {aggregator_name} not found")
