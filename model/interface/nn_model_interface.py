from abc import ABC, abstractmethod
import torch.nn as nn


class NNModel(ABC, nn.Module):
    """
    class name should be the same as file name
    """
    def __init__(self, location, split_layer):
        super(NNModel, self).__init__()
        self.cfg = self.get_config()
        assert split_layer < len(self.cfg)
        self.split_layer = split_layer
        self.location = location
        self.features, self.denses = self._make_layers(self.cfg)
        self._initialize_weights()

    def forward(self, x):
        if len(self.features) > 0:
            out = self.features(x)
        else:
            out = x
        if len(self.denses) > 0:
            out = out.view(out.size(0), -1)
            out = self.denses(out)

        return out

    @abstractmethod
    def _make_layers(self, cfg):
        """
        notice that you can change any part of the method if the input and output still matches the requirements
        :param cfg: the configuration of each layer in a list
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
        # you can refer to implemented vgg class to see an example
        pass

    @abstractmethod
    def get_config(self):
        """

        :return: the configuration of each layer in a list  of (Type, in_channels, out_channels, kernel_size, out_size(c_out*h*w), flops(c_out*h*w*k*k*c_in))
        """
        pass
