from pathlib import Path
import logging
from PIL import Image
import pandas as pd
import torch
import torch.utils.data
from torchvision import transforms


class SingleViewDataset(torch.utils.data.Dataset):
    """Single View Dataset"""

    # Inputs
    _path: Path
    _subset: str
    _transform: transforms.Compose

    # Local
    _logger: logging.Logger
    _class_names: list[str]
    _dataset: pd.DataFrame

    def __init__(
        self,
        path: Path,
        subset: str,
        transform: transforms.Compose,
    ) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        self._path = path
        self._subset = subset
        self._transform = transform

        filenames: list[Path] = list(self._path.glob("*/*/*"))
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

    def __len__(self) -> int:
        return len(self._dataset)

    @property
    def n_classes(self) -> int:
        return len(self._class_names)

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor]:
        """Return the images and the class label"""
        self._logger.debug("Loading image %i", index)
        # Get the class index
        class_index = self._dataset["class_index"].iloc[index]
        # Get the image
        filename = self._dataset["filenames"].iloc[index]
        image = Image.open(filename).convert("RGB")
        image = self._transform(image)
        return class_index, image
