"""Define the SVCNNSettings class"""

from .model_settings import ModelSettings


class SVCNNSettings(ModelSettings):
    """Settings for the SVCNN class"""

    class_names: list[str]
    n_classes: int
    pretraining: bool
    cnn_name: str
