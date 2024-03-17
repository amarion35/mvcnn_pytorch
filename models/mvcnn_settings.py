"""Define the MVCNNSettings class"""

from .model_settings import ModelSettings


class MVCNNSettings(ModelSettings):
    """Settings for the MVCNN class"""

    classes_names: list[str]
    n_classes: int
    num_views: int
