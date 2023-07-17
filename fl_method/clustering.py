from config import config


def bandwith_clustering(self, state, bandwidth):
    # sort bandwidth in config.CLIENTS_LIST order
    bandwidth_order = []
    for c in config.CLIENTS_LIST:
        bandwidth_order.append(bandwidth[c])

    labels = [0, 0, 1, 0, 0]  # Previous clustering results in RL
    for i in range(len(bandwidth_order)):
        if bandwidth_order[i] < 5:
            labels[i] = 2  # If network speed is limited under 5Mbps, we assign the device into group 2

    return labels
