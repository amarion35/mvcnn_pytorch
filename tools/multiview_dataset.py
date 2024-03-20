from pathlib import Path
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
        self._path = path
        self._set = subset
        self._transform = transform

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
        self._samples_names = list(self._dataset["model_name"].unique())

        # Get the number of views per model
        sizes = self._dataset.groupby(["class_name", "model_name"]).size()
        # Check if all models have the same number of views
        assert len(set(sizes)) == 1, "All models must have the same number of views"
        self._num_views = int(sizes.iloc[0])  # type: ignore

    def __len__(self) -> int:
        return int(len(self._filepaths) / self._num_views)

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor]:
        """Return the images and the class label"""
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
