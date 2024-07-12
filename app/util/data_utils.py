import pickle

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from app.config import config
from app.config.config import *
from app.dataset.entity.dataset_interface import DatasetInterface

DATASET_BASE_DIR = "app.dataset.entity."


def get_trainset():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset: DatasetInterface = get_class()(
        root=config.dataset_path + config.dataset_name + '/', train=True, transform=transform_train)
    return trainset


def get_trainloader(trainset, part_tr, cpu_count):
    subset = Subset(trainset, part_tr)
    trainloader = DataLoader(
        subset, batch_size=B, shuffle=True, num_workers=cpu_count)
    return trainloader


def sizeofmessage(msg):
    # size = 0
    # for i in range(len(msg)):
    #     size += sys.getsizeof(pickle.dumps(msg[i]))
    # # fed_logger.info(Fore.RED+f"{msg[0]},{size}")
    # return size * 8
    return sys.getsizeof(pickle.dumps(msg))


def get_testloader(testset, cpu_count):
    return torch.utils.data.DataLoader(testset, batch_size=B, shuffle=False, num_workers=cpu_count)


def get_testset():
    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
         ])
    return get_class()(root=config.dataset_path + config.dataset_name + '/', train=False,
                       transform=transform_test)


def get_class():
    kls = DATASET_BASE_DIR + config.dataset_name + '.' + config.dataset_name
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m
