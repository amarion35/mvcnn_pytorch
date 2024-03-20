"""Define the SVCNNSettings class"""

from .model_settings import ModelSettings


class SVCNNSettings(ModelSettings):
    """Settings for the SVCNN class"""

    pretraining: bool
    cnn_name: str
