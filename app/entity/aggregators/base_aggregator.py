from abc import abstractmethod, ABC

from app.dto.model import Model


class BaseAggregator(ABC):
    @abstractmethod
    def aggregate(self, base_model: Model, gathered_models: list[Model]) -> Model:
        pass
