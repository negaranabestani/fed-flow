import os
from typing import Callable, Optional, Any
from abc import abstractmethod
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive, check_integrity

from config import config


class DatasetInterface(VisionDataset):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False, dataset_url=None, base_folder=None,
                 filename=None):

        self.base_folder = base_folder
        self.test_list = self.get_test_list()
        self.train_list = self.get_train_list()
        self.dataset_url = dataset_url
        self.batch_size = config.B
        self.filename = filename
        self.train = train  # training set or test set
        if download:
            self.download()
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")
        self.downloaded_list = self.set_downloaded_list()
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.targets = []
        self.data: Any = self.set_data()

    def _check_integrity(self) -> bool:
        for filename, md5 in self.train_list + self.test_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.dataset_url, self.root, filename=self.filename, md5=None)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"

    @abstractmethod
    def set_downloaded_list(self):
        pass

    @abstractmethod
    def get_train_list(self) -> list:
        pass

    @abstractmethod
    def get_test_list(self) -> list:
        pass

    @abstractmethod
    def set_data(self):
        pass
