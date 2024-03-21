"""Define the DatasetSettings class"""

from pathlib import Path
from pydantic_settings import BaseSettings


class DatasetSettings(BaseSettings):
    """Settings for the multi-view CNN dataset"""

    path: Path
