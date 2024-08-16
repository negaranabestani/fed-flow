class NodeCoordinate:
    latitude: float
    longitude: float
    altitude: float
    seconds_since_start: float

    def __init__(self, latitude: float, longitude: float, altitude: float, seconds_since_start: float):
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.seconds_since_start = seconds_since_start

    def __str__(self):
        return (f"Latitude: {self.latitude}, Longitude: {self.longitude}, "
                f"Altitude: {self.altitude}, Seconds Since Start: {self.seconds_since_start}")