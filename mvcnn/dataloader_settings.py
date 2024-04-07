"""Define the DataLoaderSettings class."""

from pydantic_settings import BaseSettings


class DataloaderSettings(BaseSettings):
    """Settings for the DataLoader."""

    batch_size: int
    num_workers: int
    shuffle: bool
