from abc import abstractmethod, ABC

from app.dto.base_model import BaseModel


class BaseAggregator(ABC):
    @abstractmethod
    def aggregate(self, base_model: BaseModel, gathered_models: list[BaseModel]) -> BaseModel:
        pass
