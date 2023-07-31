from fl_method import aggregation
from fl_method import clustering
from fl_method import splitting

# a mapping of fl methods to make function call easier
fl_methods = {
    "fed_avg": aggregation.fed_avg,
    "bandwidth": clustering.bandwidth,
    "none_clustering": clustering.none,
    "none_splitting": splitting.none
}
