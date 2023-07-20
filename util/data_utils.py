import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from config import config
from config.config import *


def get_local_dataloader(CLIENT_IDEX, cpu_count):
    indices = list(range(N))
    part_tr = indices[int((N / K) * CLIENT_IDEX): int((N / K) * (CLIENT_IDEX + 1))]

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root=dataset_path, train=True, download=False, transform=transform_train)
    subset = Subset(trainset, part_tr)
    trainloader = DataLoader(
        subset, batch_size=B, shuffle=True, num_workers=cpu_count)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer',
    #            'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader


def get_testloader(testset):
    return torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)


def get_testset():
    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
         ])
    return torchvision.datasets.CIFAR10(root=config.dataset_path, train=False, download=False,
                                        transform=transform_test)
