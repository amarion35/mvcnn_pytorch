from pathlib import Path
import logging
from PIL import Image
import pandas as pd
import torch
import torch.utils.data
from torchvision import transforms


class MultiviewDataset(torch.utils.data.Dataset):
    """Multi View Dataset"""

    # Inputs
    _path: Path
    _dataset: pd.DataFrame
    _set: str
    _transform: transforms.Compose

    # Local
    _logger: logging.Logger
    _class_names: list[str]
    _samples_names: list[str]
    _num_views: int
    _filepaths: list[str]

    def __init__(
        self,
        path: Path,
        subset: str,
        transform: transforms.Compose,
    ) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        self._path = path
        self._set = subset
        self._transform = transform

        filenames = list(self._path.glob("*/*/*"))
        filenames = [f for f in filenames if not f.name.startswith(".")]
        self._class_names = sorted(list(set([f.parent.parent.name for f in filenames])))
        self._dataset = pd.DataFrame(
            {
                "filenames": filenames,
                "class_name": [f.parent.parent.name for f in filenames],
                "class_index": [
                    self._class_names.index(f.parent.parent.name) for f in filenames
                ],
                "model_name": [f.name.rsplit(".", 2)[0] for f in filenames],
                "subset": [f.parent.name for f in filenames],
            }
        )
        # Filter the dataset
        self._dataset = self._dataset[self._dataset["subset"] == subset]

        # Get the file paths
        self._samples_names = list(self._dataset["model_name"].unique())

        # Get the number of views per model
        n_views = self._dataset.groupby("model_name").size()
        # Check if all models have the same number of views
        if not n_views.nunique() == 1:
            raise ValueError("All models must have the same number of views")

        self._num_views = int(n_views.iloc[0])

    def __len__(self) -> int:
        return int(len(self._dataset) / self._num_views)

    @property
    def n_classes(self) -> int:
        return len(self._class_names)

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor]:
        """Return the images and the class label"""
        self._logger.debug("Loading image %i", index)
        # Get the model name
        model_name = self._samples_names[index]
        # Get the class name
        class_name = self._dataset[self._dataset["model_name"] == model_name].iloc[0][
            "class_name"
        ]
        # Get the class index
        class_index = self._class_names.index(class_name)
        # Get the file paths
        file_paths = self._dataset[self._dataset["model_name"] == model_name][
            "filenames"
        ].tolist()
        # Load the images
        imgs = []
        for file_path in file_paths:
            img = Image.open(file_path).convert("RGB")
            if self._transform:
                img = self._transform(img)
            imgs.append(img)
        # Stack the images
        imgs = torch.stack(imgs)
        return class_index, imgs
