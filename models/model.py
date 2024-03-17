import os
import glob
from pathlib import Path
import torch
import torch.nn as nn

from .model_settings import ModelSettings


class Model(nn.Module):

    _settings: ModelSettings

    def __init__(self, settings: ModelSettings) -> None:
        super(Model, self).__init__()
        self._settings = settings

    def save(self, path: Path, epoch: int = 0) -> None:
        complete_path = os.path.join(path, self._settings.name)
        if not os.path.exists(complete_path):
            os.makedirs(complete_path)
        torch.save(
            self.state_dict(),
            os.path.join(complete_path, f"model-{str(epoch).zfill(5)}.pth"),
        )

    def save_results(self, path: Path, data):
        raise NotImplementedError("Model subclass must implement this method.")

    def load(self, path: Path, model_file=None) -> None:
        complete_path = os.path.join(path, self._settings.name)
        if not os.path.exists(complete_path):
            raise IOError(f"{self._settings.name} directory does not exist in {path}")

        if model_file is None:
            model_files = glob.glob(complete_path + "/*")
            mf = max(model_files)
        else:
            mf = os.path.join(complete_path, model_file)

        self.load_state_dict(torch.load(mf))
