"""Defines the TrainerSettings class."""

from pathlib import Path
from pydantic_settings import BaseSettings


class TrainerSettings(BaseSettings):
    """Settings for the Trainer."""

    log_dir: Path
    steps_per_epoch: int
    epochs: int
