import logging
from pathlib import Path
import torch
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms

from tools import Trainer, MultiviewDataset, SingleViewDataset, DatasetSettings
from models import MVCNN, SVCNN, MVCNNSettings, SVCNNSettings


from pydantic_settings import BaseSettings


class OptimizerSettings(BaseSettings):
    lr: float
    betas: tuple[float, float]
    weight_decay: float


class dataloaderSettings(BaseSettings):
    batch_size: int
    num_workers: int
    shuffle: bool


class TrainerSettings(BaseSettings):
    log_dir: Path
    steps_per_epoch: int


class Settings(BaseSettings):
    name: str
    dataset_settings: DatasetSettings
    svcnn_settings: SVCNNSettings
    svcnn_optimizer_settings: OptimizerSettings
    svcnn_train_dataloader_settings: dataloaderSettings
    svcnn_val_dataloader_settings: dataloaderSettings
    svcnn_trainer_settings: TrainerSettings
    mvcnn_settings: MVCNNSettings
    mvcnn_optimizer_settings: OptimizerSettings
    mvcnn_train_dataloader_settings: dataloaderSettings
    mvcnn_val_dataloader_settings: dataloaderSettings
    mvcnn_trainer_settings: TrainerSettings


def train() -> None:

    settings_path = Path("settings.json")
    settings = Settings.model_validate_json(settings_path.read_text(encoding="utf-8"))

    svcnn = SVCNN(settings=settings.svcnn_settings)

    svcnn_optimizer = optim.Adam(
        params=svcnn.parameters(),
        lr=settings.svcnn_optimizer_settings.lr,
        betas=settings.svcnn_optimizer_settings.betas,
        weight_decay=settings.svcnn_optimizer_settings.weight_decay,
    )

    svcnn_train_transform = transforms.Compose(
        transforms=[
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    svcnn_train_dataset = SingleViewDataset(
        path=settings.dataset_settings.path,
        subset="train",
        transform=svcnn_train_transform,
    )
    svcnn_train_loader = torch.utils.data.DataLoader(
        dataset=svcnn_train_dataset,
        batch_size=settings.svcnn_train_dataloader_settings.batch_size,
        shuffle=settings.svcnn_train_dataloader_settings.shuffle,
        num_workers=settings.svcnn_train_dataloader_settings.num_workers,
    )

    svcnn_val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    svcnn_val_dataset = SingleViewDataset(
        path=settings.dataset_settings.path,
        subset="test",
        transform=svcnn_val_transform,
    )
    svcnn_val_loader = torch.utils.data.DataLoader(
        dataset=svcnn_val_dataset,
        batch_size=settings.svcnn_val_dataloader_settings.batch_size,
        shuffle=settings.svcnn_val_dataloader_settings.shuffle,
        num_workers=settings.svcnn_val_dataloader_settings.num_workers,
    )

    svcnn_trainer = Trainer(
        model=svcnn,
        train_loader=svcnn_train_loader,
        val_loader=svcnn_val_loader,
        loss=nn.CrossEntropyLoss(),
        optimizer=svcnn_optimizer,
        log_dir=settings.svcnn_trainer_settings.log_dir,
        steps_per_epoch=settings.svcnn_trainer_settings.steps_per_epoch,
    )
    svcnn_trainer.train(1)

    # STAGE 2

    mvcnn_settings = MVCNNSettings(
        name=settings.mvcnn_settings.name,
        num_views=settings.mvcnn_settings.num_views,
    )
    mvcnn = MVCNN(
        svcnn=svcnn,
        settings=mvcnn_settings,
    )

    mvcnn_optimizer = optim.Adam(
        params=mvcnn.parameters(),
        lr=settings.mvcnn_optimizer_settings.lr,
        betas=settings.mvcnn_optimizer_settings.betas,
        weight_decay=settings.mvcnn_optimizer_settings.weight_decay,
    )

    mvcnn_train_transform = transforms.Compose(
        transforms=[
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    mvcnn_train_dataset = MultiviewDataset(
        path=settings.dataset_settings.path,
        subset="train",
        transform=mvcnn_train_transform,
    )
    mvcnn_train_loader = torch.utils.data.DataLoader(
        dataset=mvcnn_train_dataset,
        batch_size=settings.mvcnn_train_dataloader_settings.batch_size,
        shuffle=settings.mvcnn_train_dataloader_settings.shuffle,
        num_workers=settings.mvcnn_train_dataloader_settings.num_workers,
    )

    mvcnn_val_dataset = MultiviewDataset(
        path=settings.dataset_settings.path,
        subset="test",
        transform=mvcnn_train_transform,
    )
    mvcnn_val_loader = torch.utils.data.DataLoader(
        dataset=mvcnn_val_dataset,
        batch_size=settings.mvcnn_val_dataloader_settings.batch_size,
        shuffle=settings.mvcnn_val_dataloader_settings.shuffle,
        num_workers=settings.mvcnn_val_dataloader_settings.num_workers,
    )

    mvcnn_trainer = Trainer(
        model=mvcnn,
        train_loader=mvcnn_train_loader,
        val_loader=mvcnn_val_loader,
        loss=nn.CrossEntropyLoss(),
        optimizer=mvcnn_optimizer,
        log_dir=settings.mvcnn_trainer_settings.log_dir,
        steps_per_epoch=settings.mvcnn_trainer_settings.steps_per_epoch,
    )
    mvcnn_trainer.train(1)


logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.getLogger("PIL").setLevel(logging.INFO)
logging.getLogger("SingleViewDataset").setLevel(logging.INFO)

if __name__ == "__main__":
    train()
