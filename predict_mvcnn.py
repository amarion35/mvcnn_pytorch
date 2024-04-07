import logging
from pathlib import Path
import torch
import torch.utils.data
from torchvision import transforms

from mvcnn import (
    MultiviewDataset,
    DatasetSettings,
    MVCNN,
    SVCNN,
    DataloaderSettings,
    OptimizerSettings,
    TrainerSettings,
    Predictor,
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
    dataset_settings: DatasetSettings
    svcnn_weights_path: Path
    mvcnn_weights_path: Path
    batch_size: int
    num_workers: int
    device: str
    output_path: Path


def predict(settings_path: Path = Path("predict_mvcnn_settings.json")) -> None:
    """Predict the classes of the test set using the MVCNN model."""

    settings = Settings.model_validate_json(settings_path.read_text(encoding="utf-8"))

    transform = transforms.Compose(
        transforms=[
            transforms.ToTensor(),
        ]
    )

    dataset = MultiviewDataset(
        path=settings.dataset_settings.path,
        subset="test",
        transform=transform,
    )
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
    )

    svcnn = SVCNN(
        cnn_name=settings.cnn_name,
        pretraining=False,
        n_classes=dataset.n_classes,
    )
    svcnn.load(settings.svcnn_weights_path)

    mvcnn = MVCNN(
        svcnn=svcnn,
        n_classes=dataset.n_classes,
    )
    mvcnn.load(settings.mvcnn_weights_path)

    predictor = Predictor(model=mvcnn, loader=loader, device=settings.device)
    pred = predictor.predict()

    # Save the predictions
    pred.to_csv(settings.output_path, index_label="indices")


logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.getLogger("PIL").setLevel(logging.INFO)
logging.getLogger("SingleViewDataset").setLevel(logging.INFO)
logging.getLogger("MultiviewDataset").setLevel(logging.INFO)

if __name__ == "__main__":
    predict()
