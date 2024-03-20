"""Define the MVCNNSettings class"""

from .model_settings import ModelSettings


class MVCNNSettings(ModelSettings):
    """Settings for the MVCNN class"""

    num_views: int
