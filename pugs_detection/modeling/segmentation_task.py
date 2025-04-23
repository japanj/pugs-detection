from torchgeo.trainers import SemanticSegmentationTask
import segmentation_models_pytorch as smp
import torch.nn as nn
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics import Accuracy, JaccardIndex, MetricCollection
from typing import Any

class CustomSegmentationTask(SemanticSegmentationTask):
    def __init__(self, **kwargs):
        # Remove 'ignore' parameter if it exists in kwargs
        if "ignore" in kwargs:
            del kwargs["ignore"]
        super().__init__(**kwargs)

    def configure_losses(self) -> None:
        """Initialize the loss criterion."""
        ignore_index: int | None = self.hparams["ignore_index"]
        match self.hparams["loss"]:
            case "ce":
                ignore_value = -1000 if ignore_index is None else ignore_index
                self.criterion: nn.Module = nn.CrossEntropyLoss(
                    ignore_index=ignore_value, weight=self.hparams["class_weights"]
                )
            case "bce":
                self.criterion = nn.BCEWithLogitsLoss()
            case "jaccard":
                self.criterion = smp.losses.JaccardLoss(mode="binary")
            case "focal":
                self.criterion = smp.losses.FocalLoss(
                    "binary", ignore_index=ignore_index, normalized=True
                )

    def configure_metrics(self) -> None:
        """Initialize the performance metrics.

        * :class:`~torchmetrics.Accuracy`: Overall accuracy
          (OA) using 'micro' averaging. The number of true positives divided by the
          dataset size. Higher values are better.
        * :class:`~torchmetrics.JaccardIndex`: Intersection
          over union (IoU). Uses 'micro' averaging. Higher valuers are better.

        .. note::
           * 'Micro' averaging suits overall performance evaluation but may not reflect
             minority class accuracy.
           * 'Macro' averaging, not used here, gives equal weight to each class, useful
             for balanced performance assessment across imbalanced classes.
        """
        kwargs = {
            "task": "binary",
            "num_classes": self.hparams["num_classes"],
            "num_labels": None,
            "ignore_index": self.hparams["ignore_index"],
        }
        metrics = MetricCollection(
            [
                Accuracy(multidim_average="global", average="micro", **kwargs),
                JaccardIndex(average="micro", **kwargs),
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def training_step(self, batch: Any) -> Tensor:
        """Compute the training loss and additional metrics.

        Args:
            batch: The output of your DataLoader.

        Returns:
            The loss tensor.
        """
        x = batch["image"]
        y = batch["mask"]
        batch_size = x.shape[0]
        y_hat = self(x).squeeze(1)
        self.train_metrics(y_hat, y)
        self.log_dict(self.train_metrics, batch_size=batch_size)

        if self.hparams["loss"] == "bce":
            y = y.float()

        loss: Tensor = self.criterion(y_hat, y)
        self.log("train_loss", loss, batch_size=batch_size)

        return loss

    def validation_step(self, batch: Any) -> None:
        """Compute the validation loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
        """
        x = batch["image"]
        y = batch["mask"]
        batch_size = x.shape[0]
        y_hat = self(x).squeeze(1)
        self.val_metrics(y_hat, y)
        self.log_dict(self.val_metrics, batch_size=batch_size)

        if self.hparams["loss"] == "bce":
            y = y.float()

        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, batch_size=batch_size)

    def test_step(self, batch: Any) -> None:
        """Compute the test loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
        """
        x = batch["image"]
        y = batch["mask"]
        batch_size = x.shape[0]
        y_hat = self(x).squeeze(1)
        self.test_metrics(y_hat, y)
        self.log_dict(self.test_metrics, batch_size=batch_size)

        if self.hparams["loss"] == "bce":
            y = y.float()

        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss, batch_size=batch_size)