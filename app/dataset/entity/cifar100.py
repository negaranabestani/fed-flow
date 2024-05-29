import os.path
import pickle
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torchvision.datasets
from PIL import Image
from torchvision.datasets.utils import check_integrity

from app.dataset.entity.dataset_interface import DatasetInterface


class cifar10(DatasetInterface):
    torchvision.datasets.CIFAR100
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test_app set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar10.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:

        super().__init__(root,
                         train,
                         transform,
                         target_transform,
                         self.url, self.base_folder,
                         self.filename)

        self._load_meta()

    def set_downloaded_list(self):
        if self.train:
            self.downloaded_list = self.train_list
        else:
            self.downloaded_list = self.test_list

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError("Dataset metadata file not found or corrupted. You can use download=True to download it")
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def get_train_list(self) -> list:
        return [
            ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
            ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
            ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
            ["data_batch_4", "634d18415352ddfa80567beed471001a"],
            ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"]
        ]

    def get_test_list(self) -> list:
        return [
            ["test_batch", "40351d587109b95175f43aff81a1287e"]
        ]

    def set_data(self):
        # now load the picked numpy arrays
        for file_name, checksum in self.downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        data = np.vstack(self.data).reshape(-1, 3, 32, 32)

        data = data.transpose((0, 2, 3, 1))  # convert to HWC
        return data
