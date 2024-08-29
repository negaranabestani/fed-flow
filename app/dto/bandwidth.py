class BandWidth:
    bandwidth: float

    def __init__(self, transferred_bytes: float, time: float):
        self.bandwidth = transferred_bytes / time

    def __eq__(self, other):
        return self.bandwidth == other.bandwidth

    def __hash__(self):
        return hash(self.bandwidth)
