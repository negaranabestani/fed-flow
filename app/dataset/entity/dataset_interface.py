import os
from abc import abstractmethod
from typing import Callable, Optional, Any

from torchvision.datasets.utils import download_and_extract_archive, check_integrity
from torchvision.datasets.vision import VisionDataset

from app.config import config


class DatasetInterface(VisionDataset):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 dataset_url=None, base_folder=None,
                 filename=None):
        """
        Args:
            root: the path to the folder
            train: true if the data partition is used for training the model
            download: true if the dataset does not already exist
            dataset_url: dataset class name should be the same as folder name that holds the actual data
            base_folder: the last inner folder that holds actual data
            filename: the final downloaded file name
        """

        self.base_folder = base_folder
        self.test_list = None
        self.train_list = None
        self.dataset_url = dataset_url
        self.batch_size = config.B
        self.filename = filename
        self.train = train  # training set or test_app set
        # self.is_download = download
        self.downloaded_list = None
        self.targets = []
        self.data: Any = []
        super().__init__(root=root, transform=transform, target_transform=target_transform)
        self.get_dataset()

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

    def get_dataset(self):
        self.test_list = self.get_test_list()
        self.train_list = self.get_train_list()
        if not self._check_integrity():
            self.download()
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")
        self.set_downloaded_list()
        self.data: Any = self.set_data()
