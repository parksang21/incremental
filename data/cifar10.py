from torchvision import transforms
from torchvision.datasets import CIFAR10
from typing import Any, Callable, Optional
import numpy as np
from PIL import Image

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_valid = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

general_transform = {
    'train': transform_train,
    'test': transform_valid
}


class SplitCifar10(CIFAR10):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 ) -> None:

        super(SplitCifar10, self).__init__(root, train=train, transform=transform,
                                           target_transform=target_transform, download=download)

        self.split_idx = None

    def set_split(self, split):
        np_target = np.array(self.targets)
        split_idxs = np.isin(np_target, split)
        np_target = np.where(split_idxs == True)[0]

        self.split_idx = np_target

    def __getitem__(self, index):
        assert self.split_idx is not None
        index = self.split_idx[index]

        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        assert self.split_idx is not None

        return len(self.split_idx)
