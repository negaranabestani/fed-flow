from abc import ABC, abstractmethod
import torch.nn as nn


# ToDo add description about each method
class NNModel(ABC, nn.Module):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def _make_layers(self, cfg):
        """
        notice that you can change any part of the method if the input and output still matches the requirements
        :param cfg: the configuration of each layer in a list -> # (Type, in_channels, out_channels, kernel_size, out_size(c_out*h*w), flops(c_out*h*w*k*k*c_in))
        :return: nn.Sequential(*features), nn.Sequential(*denses)
        """
        features = []
        denses = []
        if self.location == 'Server':
            cfg = cfg[self.split_layer + 1:]

        if self.location == 'Client':
            cfg = cfg[:self.split_layer + 1]

        if self.location == 'Unit':  # Get the holistic nn_model
            pass

        #  TODO fill the featured and dense based on your model configuration
        nn.Sequential(*features), nn.Sequential(*denses)

    @abstractmethod
    def _initialize_weights(self):
        pass
