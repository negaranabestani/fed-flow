from enum import Enum


class NodeType(Enum):
    CLIENT = "client"
    EDGE = "edge"
    SERVER = "server"

    @classmethod
    def from_value(cls, value: str) -> 'NodeType':
        """Get the NodeType enum member from its value"""
        value = value.lower()
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"No NodeType found for value '{value}'")
