import os
from pathlib import Path
import shutil
import typer
import torch
import torch.utils
import torch.optim as optim
import torch.nn as nn

from tools import Trainer, MultiviewDataset, SingleViewDataset
from models import MVCNN, SVCNN, MVCNNSettings, SVCNNSettings


def create_folder(log_dir):
    # make summary folder
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        print("WARNING: summary folder already exists!! It will be overwritten!!")
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)


from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    name: str
    svcnn_settings: SVCNNSettings
    mvcnn_settings: MVCNNSettings


def train(
    # name="MVCNN",
    # batch_size=8,
    # num_models=1000,
    # lr=5e-5,
    # weight_decay=0.0,
    # no_pretraining=False,
    # cnn_name="vgg11",
    # num_views=12,
    # train_path="modelnet40_images_new_12x/*/train",
    # val_path="modelnet40_images_new_12x/*/test",
) -> None:

    settings_path = Path("settings.json")
    settings = Settings.model_validate_json(settings_path.read_text())

    pretraining = not no_pretraining

    classes_names: list[str] = [
        "airplane",
        "bathtub",
        "bed",
        "bench",
        "bookshelf",
        "bottle",
        "bowl",
        "car",
        "chair",
        "cone",
        "cup",
        "curtain",
        "desk",
        "door",
        "dresser",
        "flower_pot",
        "glass_box",
        "guitar",
        "keyboard",
        "lamp",
        "laptop",
        "mantel",
        "monitor",
        "night_stand",
        "person",
        "piano",
        "plant",
        "radio",
        "range_hood",
        "sink",
        "sofa",
        "stairs",
        "stool",
        "table",
        "tent",
        "toilet",
        "tv_stand",
        "vase",
        "wardrobe",
        "xbox",
    ]
    n_classes = len(classes_names)

    svcnn = SVCNN(settings=settings.svcnn_settings)

    svcnn_optimizer = optim.Adam(svcnn.parameters(), lr=lr, weight_decay=weight_decay)

    n_models_train = num_models * num_views

    svcnn_train_dataset = SingleViewDataset(
        train_path,
        scale_aug=False,
        rot_aug=False,
        num_models=n_models_train,
        num_views=num_views,
    )
    svcnn_train_loader = torch.utils.data.DataLoader(
        svcnn_train_dataset, batch_size=64, shuffle=True, num_workers=0
    )

    svcnn_val_dataset = SingleViewDataset(
        val_path, scale_aug=False, rot_aug=False, test_mode=True
    )
    svcnn_val_loader = torch.utils.data.DataLoader(
        svcnn_val_dataset, batch_size=64, shuffle=False, num_workers=0
    )

    svcnn_trainer = Trainer(
        svcnn,
        svcnn_train_loader,
        svcnn_val_loader,
        svcnn_optimizer,
        nn.CrossEntropyLoss(),
        "svcnn",
        log_dir,
        num_views=1,
    )
    svcnn_trainer.train(1)

    # STAGE 2
    log_dir = name + "_stage_2"
    create_folder(log_dir)
    mvcnn_settings = MVCNNSettings(
        name=name,
        class_names=classes_names,
        n_classes=40,
        num_views=num_views,
    )
    mvcnn = MVCNN(
        model=svcnn,
        settings=mvcnn_settings,
    )
    del svcnn

    svcnn_optimizer = optim.Adam(
        mvcnn.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )

    svcnn_train_dataset = MultiviewDataset(
        train_path,
        scale_aug=False,
        rot_aug=False,
        num_models=n_models_train,
        num_views=num_views,
    )
    svcnn_train_loader = torch.utils.data.DataLoader(
        svcnn_train_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )  # shuffle needs to be false! it's done within the trainer

    svcnn_val_dataset = MultiviewDataset(
        val_path, scale_aug=False, rot_aug=False, num_views=num_views
    )
    svcnn_val_loader = torch.utils.data.DataLoader(
        svcnn_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    print("num_train_files: " + str(len(svcnn_train_dataset._filepaths)))
    print("num_val_files: " + str(len(svcnn_val_dataset._filepaths)))
    svcnn_trainer = Trainer(
        mvcnn,
        svcnn_train_loader,
        svcnn_val_loader,
        svcnn_optimizer,
        nn.CrossEntropyLoss(),
        "mvcnn",
        log_dir,
        num_views=num_views,
    )
    svcnn_trainer.train(1)


if __name__ == "__main__":
    train()
