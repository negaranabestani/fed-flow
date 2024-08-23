from dataclasses import dataclass


@dataclass(eq=True, frozen=True)
class NodeIdentifier:
    ip: str
    port: int

    def __str__(self):
        return f"{self.ip}:{self.port}"

    def get_exchange_name(self):
        return self.__str__()
