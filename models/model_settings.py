"""Define the ModelSettings class"""

from pydantic_settings import BaseSettings


class ModelSettings(BaseSettings):
    """Settings for the Model class"""

    name: str
