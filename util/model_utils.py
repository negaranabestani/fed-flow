import argparse
import collections

import numpy as np
import torch
import torch.nn.init as init
import tqdm

from config import config
from config.logger import fed_logger
from entity.nn_model.vgg import *

np.random.seed(0)
torch.manual_seed(0)


def get_model(location, model_name, layer, device, cfg):
    cfg = cfg.copy()
    net = VGG(location, model_name, layer, cfg)
    net = net.to(device)
    fed_logger.debug(str(net))
    return net


def split_weights_client(weights, cweights):
    for key in cweights:
        assert cweights[key].size() == weights[key].size()
        cweights[key] = weights[key]
    return cweights


def split_weights_server(weights, cweights, sweights):
    ckeys = list(cweights)
    skeys = list(sweights)
    keys = list(weights)

    for i in range(len(skeys)):
        assert sweights[skeys[i]].size() == weights[keys[i + len(ckeys)]].size()
        sweights[skeys[i]] = weights[keys[i + len(ckeys)]]

    return sweights


def concat_weights(weights, cweights, sweights):
    concat_dict = collections.OrderedDict()

    ckeys = list(cweights)
    skeys = list(sweights)
    keys = list(weights)

    for i in range(len(ckeys)):
        concat_dict[keys[i]] = cweights[ckeys[i]]

    for i in range(len(skeys)):
        concat_dict[keys[i + len(ckeys)]] = sweights[skeys[i]]

    return concat_dict


def zero_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.zeros_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            init.zeros_(m.weight)
            init.zeros_(m.bias)
            init.zeros_(m.running_mean)
            init.zeros_(m.running_var)
        elif isinstance(m, nn.Linear):
            init.zeros_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
    return net


def norm_list(alist):
    return [l / sum(alist) for l in alist]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def test(uninet, testloader, device, criterion):
    uninet.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(testloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = uninet(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    fed_logger.info('Test Accuracy: {}'.format(acc))

    # Save checkpoint.
    torch.save(uninet.state_dict(), './' + config.model_name + '.pth')

    return acc
