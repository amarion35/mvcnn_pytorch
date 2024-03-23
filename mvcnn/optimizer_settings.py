"""Defines the OptimizerSettings class."""

from pydantic_settings import BaseSettings


class OptimizerSettings(BaseSettings):
    """Settings for the optimizer."""

    lr: float
    betas: tuple[float, float]
    weight_decay: float
