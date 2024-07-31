from app.fl_method import clustering, splitting, aggregation_method

# a mapping of fl methods to make function call easier
fl_methods = {
    "fed_avg": aggregation_method.fed_avg,
    "bandwidth": clustering.bandwidth,
    "none_clustering": clustering.none,
    "none_splitting": splitting.none,
    "fake_splitting": splitting.fake,
    "no_splitting": splitting.no_splitting,
    "no_edge_fake_splitting": splitting.no_edge_fake,
    "no_edge_rl_splitting": splitting.rl_splitting,
    "edge_based_energy_aware_rl_splitting": splitting.edge_based_energy_aware_rl_splitting,
    "only_edge_splitting": splitting.only_edge_splitting,
    "only_server_splitting": splitting.only_server_splitting,
    "random_splitting": splitting.randomSplitting,
    "fedmec_splitting": splitting.FedMec,
}
