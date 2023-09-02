from app.fl_method import aggregation, clustering, splitting

# a mapping of fl methods to make function call easier
fl_methods = {
    "fed_avg": aggregation.fed_avg,
    "bandwidth": clustering.bandwidth,
    "none_clustering": clustering.none,
    "none_splitting": splitting.none,
    "fake_splitting": splitting.fake,
    "no_splitting": splitting.no_splitting
}
