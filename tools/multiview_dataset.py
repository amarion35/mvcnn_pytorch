from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torchvision import transforms


class MultiviewDataset(torch.utils.data.Dataset):

    _class_names: list[str]
    _path: Path
    _dataset: pd.DataFrame
    _set: str
    _scale_aug: bool
    _rot_aug: bool
    _test_mode: bool
    _num_views: int
    _filepaths: list[str]

    def __init__(
        self,
        path: Path,
        subset: str,
        scale_aug: bool = False,
        rot_aug: bool = False,
        test_mode: bool = False,
    ) -> None:
        self._path = path
        self._set = subset
        self._scale_aug = scale_aug
        self._rot_aug = rot_aug
        self._test_mode = test_mode

        filenames = list(self._path.glob("*/*/*"))
        self._class_names = sorted(list(set([f.parent.parent.name for f in filenames])))
        self._dataset = pd.DataFrame(
            {
                "filenames": filenames,
                "class_name": [f.parent.parent.name for f in filenames],
                "class_index": [
                    self._class_names.index(f.parent.parent.name) for f in filenames
                ],
                "model_name": [f.parent.name.rsplit(".", 2)[0] for f in filenames],
                "subset": [f.parent.name for f in filenames],
            }
        )

        size = self._dataset.groupby(["class_name", "model_name"]).size()

        if self._test_mode:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __len__(self) -> int:
        return int(len(self._filepaths) / self._num_views)

    # def __getitem__(self, idx: int):
    #     path = self._filepaths[idx * self._num_views]
    #     class_name = path.split("/")[-3]
    #     class_id = self._class_names.index(class_name)
    #     # Use PIL instead
    #     imgs = []
    #     for i in range(self._num_views):
    #         im = Image.open(self._filepaths[idx * self._num_views + i]).convert("RGB")
    #         if self.transform:
    #             im = self.transform(im)
    #         imgs.append(im)

    #     return (
    #         class_id,
    #         torch.stack(imgs),
    #         self._filepaths[idx * self._num_views : (idx + 1) * self._num_views],
    #     )

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor]:
        """Return the images and the class label"""
        row = self._dataset.iloc[index]
        class_id = row["class_index"]
        images = []
