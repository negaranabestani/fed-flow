import aggregation
import clustering
# a mapping of fl methods to make function call easier
fl_methods = {
    "fed_avg": aggregation.fed_avg,
    "bandwith_clustering": clustering.bandwith_clustering
}
