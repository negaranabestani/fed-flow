import aggregation
import clustering
# a mapping of fl methods to make function call easier
fl_methods = {
    "fed_avg": aggregation.fed_avg,
    "bandwidth_clustering": clustering.bandwidth_clustering
}
