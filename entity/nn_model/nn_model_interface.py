from abc import ABC, abstractmethod
import torch.nn as nn


class NNModel(ABC, nn.Module):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def _make_layers(self, cfg):
        pass

    @abstractmethod
    def _initialize_weights(self):
        pass
