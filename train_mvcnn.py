import logging
from pathlib import Path
import torch
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms

from mvcnn import (
    Trainer,
    MultiviewDataset,
    SingleViewDataset,
    DatasetSettings,
    Metrics,
    Accuracy,
    MVCNN,
    SVCNN,
    DataloaderSettings,
    OptimizerSettings,
    TrainerSettings,
    RandomDiscreetRotation,
)


from pydantic_settings import BaseSettings


class ModelSettings(BaseSettings):
    """Settings for the training process."""

    optimizer_settings: OptimizerSettings
    train_dataloader_settings: DataloaderSettings
    val_dataloader_settings: DataloaderSettings
    trainer_settings: TrainerSettings


class Settings(BaseSettings):
    """Settings for the training process."""

    name: str
    cnn_name: str
    pretraining: bool
    dataset_settings: DatasetSettings
    svcnn_model_settings: ModelSettings
    mvcnn_model_settings: ModelSettings


def train_svcnn(settings: Settings) -> SVCNN:

    svcnn_train_transform = transforms.Compose(
        transforms=[
            transforms.ToTensor(),
            RandomDiscreetRotation(degrees=[0, 90, 180, 270]),
        ]
    )

    svcnn_val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    svcnn_train_dataset = SingleViewDataset(
        path=settings.dataset_settings.path,
        subset="train",
        transform=svcnn_train_transform,
    )
    svcnn_train_loader = torch.utils.data.DataLoader(
        dataset=svcnn_train_dataset,
        batch_size=settings.svcnn_model_settings.train_dataloader_settings.batch_size,
        shuffle=settings.svcnn_model_settings.train_dataloader_settings.shuffle,
        num_workers=settings.svcnn_model_settings.train_dataloader_settings.num_workers,
    )
    svcnn_val_dataset = SingleViewDataset(
        path=settings.dataset_settings.path,
        subset="test",
        transform=svcnn_val_transform,
    )
    svcnn_val_loader = torch.utils.data.DataLoader(
        dataset=svcnn_val_dataset,
        batch_size=settings.svcnn_model_settings.val_dataloader_settings.batch_size,
        shuffle=settings.svcnn_model_settings.val_dataloader_settings.shuffle,
        num_workers=settings.svcnn_model_settings.val_dataloader_settings.num_workers,
    )

    svcnn = SVCNN(
        cnn_name=settings.cnn_name,
        pretraining=settings.pretraining,
        n_classes=svcnn_train_dataset.n_classes,
    )

    svcnn_optimizer = optim.Adam(
        params=svcnn.parameters(),
        lr=settings.svcnn_model_settings.optimizer_settings.lr,
        betas=settings.svcnn_model_settings.optimizer_settings.betas,
        weight_decay=settings.svcnn_model_settings.optimizer_settings.weight_decay,
    )

    svcnn_metrics = Metrics(metrics=[Accuracy()])

    svcnn_trainer = Trainer(
        model=svcnn,
        train_loader=svcnn_train_loader,
        val_loader=svcnn_val_loader,
        loss=nn.CrossEntropyLoss(),
        metrics=svcnn_metrics,
        optimizer=svcnn_optimizer,
        log_dir=settings.svcnn_model_settings.trainer_settings.log_dir,
        steps_per_epoch=settings.svcnn_model_settings.trainer_settings.steps_per_epoch,
        device=settings.svcnn_model_settings.trainer_settings.device,
    )

    svcnn_trainer.train(settings.svcnn_model_settings.trainer_settings.epochs)

    return svcnn


def train_mvcnn(svcnn: SVCNN, settings: Settings) -> MVCNN:

    mvcnn_train_transform = transforms.Compose(
        transforms=[
            transforms.ToTensor(),
            RandomDiscreetRotation(degrees=[0, 90, 180, 270]),
        ]
    )
    mvcnn_val_transform = transforms.Compose(
        transforms=[
            transforms.ToTensor(),
        ]
    )

    mvcnn_train_dataset = MultiviewDataset(
        path=settings.dataset_settings.path,
        subset="train",
        transform=mvcnn_train_transform,
    )
    mvcnn_train_loader = torch.utils.data.DataLoader(
        dataset=mvcnn_train_dataset,
        batch_size=settings.mvcnn_model_settings.train_dataloader_settings.batch_size,
        shuffle=settings.mvcnn_model_settings.train_dataloader_settings.shuffle,
        num_workers=settings.mvcnn_model_settings.train_dataloader_settings.num_workers,
    )

    mvcnn_val_dataset = MultiviewDataset(
        path=settings.dataset_settings.path,
        subset="test",
        transform=mvcnn_val_transform,
    )
    mvcnn_val_loader = torch.utils.data.DataLoader(
        dataset=mvcnn_val_dataset,
        batch_size=settings.mvcnn_model_settings.val_dataloader_settings.batch_size,
        shuffle=settings.mvcnn_model_settings.val_dataloader_settings.shuffle,
        num_workers=settings.mvcnn_model_settings.val_dataloader_settings.num_workers,
    )

    mvcnn = MVCNN(
        svcnn=svcnn,
        n_classes=mvcnn_train_dataset.n_classes,
    )

    mvcnn_optimizer = optim.Adam(
        params=mvcnn.parameters(),
        lr=settings.mvcnn_model_settings.optimizer_settings.lr,
        betas=settings.mvcnn_model_settings.optimizer_settings.betas,
        weight_decay=settings.mvcnn_model_settings.optimizer_settings.weight_decay,
    )

    mvcnn_metrics = Metrics(metrics=[Accuracy()])

    mvcnn_trainer = Trainer(
        model=mvcnn,
        train_loader=mvcnn_train_loader,
        val_loader=mvcnn_val_loader,
        loss=nn.CrossEntropyLoss(),
        metrics=mvcnn_metrics,
        optimizer=mvcnn_optimizer,
        log_dir=settings.mvcnn_model_settings.trainer_settings.log_dir,
        steps_per_epoch=settings.mvcnn_model_settings.trainer_settings.steps_per_epoch,
        device=settings.mvcnn_model_settings.trainer_settings.device,
    )

    mvcnn_trainer.train(settings.mvcnn_model_settings.trainer_settings.epochs)

    return mvcnn


def train() -> None:

    settings_path = Path("train_mvcnn_settings.json")
    settings = Settings.model_validate_json(settings_path.read_text(encoding="utf-8"))

    # STAGE 1
    svcnn = train_svcnn(settings)

    # STAGE 2
    mvcnn = train_mvcnn(svcnn, settings)


logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.getLogger("PIL").setLevel(logging.INFO)
logging.getLogger("SingleViewDataset").setLevel(logging.INFO)
logging.getLogger("MultiviewDataset").setLevel(logging.INFO)

if __name__ == "__main__":
    train()
