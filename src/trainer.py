
"""
Research-grade training manager for NeuroBrain.

Handles:
    - training epochs
    - sliding-window validation
    - patient-level Dice metrics
    - checkpointing
    - training history
"""

import json
import os

import torch

from src.inference import predict_volume
from src.metrics import multiclass_dice, tumor_region_dice


class Trainer:
    """
    Training manager for the NeuroBrain 3D U-Net.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        scheduler=None,
        checkpoint_dir="results/checkpoints",
        roi_size=(96, 96, 96),
        sw_batch_size=1,
        overlap=0.25,
    ):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

        self.checkpoint_dir = checkpoint_dir

        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap

        os.makedirs(
            self.checkpoint_dir,
            exist_ok=True,
        )

        self.best_mean_tumor_dice = -1.0

        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "class_0_dice": [],
            "class_1_dice": [],
            "class_2_dice": [],
            "class_3_dice": [],
            "WT_dice": [],
            "TC_dice": [],
            "ET_dice": [],
            "mean_tumor_dice": [],
            "learning_rate": [],
        }

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    def train_one_epoch(self):

        self.model.train()

        running_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:

            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad(
                set_to_none=True
            )

            predictions = self.model(images)

            loss = self.criterion(
                predictions,
                labels,
            )

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        if num_batches == 0:
            return 0.0

        return running_loss / num_batches

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    @torch.no_grad()
    def validate(self):

        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        # Patient-level metric storage.
        class_scores = {
            0: [],
            1: [],
            2: [],
            3: [],
        }

        region_scores = {
            "WT": [],
            "TC": [],
            "ET": [],
        }

        for batch in self.val_loader:

            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            # -------------------------------------------------
            # Full-volume sliding-window inference
            # -------------------------------------------------

            predictions = predict_volume(
                model=self.model,
                image=images,
                roi_size=self.roi_size,
                sw_batch_size=self.sw_batch_size,
                overlap=self.overlap,
            )

            # Validation loss.
            loss = self.criterion(
                predictions,
                labels,
            )

            total_loss += loss.item()
            num_batches += 1

            # Convert logits to segmentation labels.
            pred_labels = torch.argmax(
                predictions,
                dim=1,
            )

            # -------------------------------------------------
            # Patient-level metrics
            # -------------------------------------------------

            batch_size = pred_labels.shape[0]

            for patient_index in range(batch_size):

                patient_prediction = pred_labels[
                    patient_index:patient_index + 1
                ]

                patient_target = labels[
                    patient_index:patient_index + 1
                ]

                # Class-wise Dice.
                patient_class_dice = multiclass_dice(
                    patient_prediction,
                    patient_target,
                    num_classes=4,
                )

                for class_id in range(4):

                    class_scores[class_id].append(
                        float(
                            patient_class_dice[class_id]
                        )
                    )

                # WT / TC / ET Dice.
                patient_region_dice = tumor_region_dice(
                    patient_prediction,
                    patient_target,
                )

                for region in (
                    "WT",
                    "TC",
                    "ET",
                ):

                    region_scores[region].append(
                        float(
                            patient_region_dice[region]
                        )
                    )

        # -----------------------------------------------------
        # Aggregate validation results
        # -----------------------------------------------------

        if num_batches == 0:

            return {
                "val_loss": 0.0,
                "class_dice": {
                    0: 0.0,
                    1: 0.0,
                    2: 0.0,
                    3: 0.0,
                },
                "tumor_dice": {
                    "WT": 0.0,
                    "TC": 0.0,
                    "ET": 0.0,
                },
                "mean_tumor_dice": 0.0,
            }

        val_loss = (
            total_loss / num_batches
        )

        # Mean across patients.
        mean_class_dice = {}

        for class_id in range(4):

            scores = class_scores[class_id]

            mean_class_dice[class_id] = (
                sum(scores) / len(scores)
                if scores
                else 0.0
            )

        mean_tumor_dice = {}

        for region in (
            "WT",
            "TC",
            "ET",
        ):

            scores = region_scores[region]

            mean_tumor_dice[region] = (
                sum(scores) / len(scores)
                if scores
                else 0.0
            )

        mean_tumor = (
            mean_tumor_dice["WT"]
            + mean_tumor_dice["TC"]
            + mean_tumor_dice["ET"]
        ) / 3.0

        return {
            "val_loss": val_loss,
            "class_dice": mean_class_dice,
            "tumor_dice": mean_tumor_dice,
            "mean_tumor_dice": mean_tumor,
        }

    # ---------------------------------------------------------
    # Checkpointing
    # ---------------------------------------------------------

    def save_latest_checkpoint(
        self,
        epoch,
        train_loss,
        validation,
    ):
        """
        Save the latest model state after every epoch.
        """

        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            "latest.pt",
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": validation["val_loss"],
            "class_dice": validation["class_dice"],
            "tumor_dice": validation["tumor_dice"],
            "mean_tumor_dice": validation["mean_tumor_dice"],
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

        torch.save(
            checkpoint,
            checkpoint_path,
        )

        return checkpoint_path

    def save_checkpoint(
        self,
        epoch,
        train_loss,
        validation,
    ):

        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            "best_model.pt",
        )

        checkpoint = {
            "epoch": epoch,

            "model_state_dict":
                self.model.state_dict(),

            "optimizer_state_dict":
                self.optimizer.state_dict(),

            "train_loss":
                train_loss,

            "val_loss":
                validation["val_loss"],

            "class_dice":
                validation["class_dice"],

            "tumor_dice":
                validation["tumor_dice"],

            "mean_tumor_dice":
                validation["mean_tumor_dice"],
        }

        torch.save(
            checkpoint,
            checkpoint_path,
        )

        return checkpoint_path

    # ---------------------------------------------------------
    # History
    # ---------------------------------------------------------

    def update_history(
        self,
        epoch,
        train_loss,
        validation,
    ):

        class_dice = validation[
            "class_dice"
        ]

        tumor_dice = validation[
            "tumor_dice"
        ]

        self.history["epoch"].append(
            epoch
        )

        self.history["learning_rate"].append(
            self.optimizer.param_groups[0]["lr"]
        )

        self.history["train_loss"].append(
            train_loss
        )

        self.history["val_loss"].append(
            validation["val_loss"]
        )

        for class_id in range(4):

            self.history[
                f"class_{class_id}_dice"
            ].append(
                class_dice[class_id]
            )

        self.history["WT_dice"].append(
            tumor_dice["WT"]
        )

        self.history["TC_dice"].append(
            tumor_dice["TC"]
        )

        self.history["ET_dice"].append(
            tumor_dice["ET"]
        )

        self.history[
            "mean_tumor_dice"
        ].append(
            validation[
                "mean_tumor_dice"
            ]
        )

    def save_history(
        self,
        filename="results/training_history.json",
    ):

        directory = os.path.dirname(
            filename
        )

        if directory:
            os.makedirs(
                directory,
                exist_ok=True,
            )

        with open(
            filename,
            "w",
        ) as file:

            json.dump(
                self.history,
                file,
                indent=4,
            )

    # ---------------------------------------------------------
    # Full training
    # ---------------------------------------------------------

    def fit(
        self,
        num_epochs,
    ):

        for epoch in range(
            1,
            num_epochs + 1,
        ):

            print()
            print("=" * 60)
            print(
                f"Epoch {epoch}/{num_epochs}"
            )
            print("=" * 60)

            train_loss = (
                self.train_one_epoch()
            )

            print(
                f"Training loss : "
                f"{train_loss:.4f}"
            )

            validation = self.validate()

            # Update learning rate using the primary
            # validation metric: mean tumor Dice.
            if self.scheduler is not None:
                self.scheduler.step(
                    validation["mean_tumor_dice"]
                )

            print(
                f"Validation loss : "
                f"{validation['val_loss']:.4f}"
            )

            print()
            print(
                "Class-wise Dice:"
            )

            for class_id in range(4):

                print(
                    f"Class {class_id}: "
                    f"{validation['class_dice'][class_id]:.4f}"
                )

            print()
            print(
                "Tumor-region Dice:"
            )

            print(
                f"WT: "
                f"{validation['tumor_dice']['WT']:.4f}"
            )

            print(
                f"TC: "
                f"{validation['tumor_dice']['TC']:.4f}"
            )

            print(
                f"ET: "
                f"{validation['tumor_dice']['ET']:.4f}"
            )

            print(
                f"\nMean tumor Dice: "
                f"{validation['mean_tumor_dice']:.4f}"
            )

            self.update_history(
                epoch,
                train_loss,
                validation,
            )

            # Always save the latest state so the experiment
            # can be resumed after a Colab interruption.
            latest_path = self.save_latest_checkpoint(
                epoch,
                train_loss,
                validation,
            )

            print()
            print("Latest checkpoint saved:")
            print(latest_path)

            # Save best model based on
            # mean tumor-region Dice.
            if (
                validation[
                    "mean_tumor_dice"
                ]
                > self.best_mean_tumor_dice
            ):

                self.best_mean_tumor_dice = (
                    validation[
                        "mean_tumor_dice"
                    ]
                )

                checkpoint_path = (
                    self.save_checkpoint(
                        epoch,
                        train_loss,
                        validation,
                    )
                )

                print()
                print(
                    "New best model saved:"
                )

                print(
                    checkpoint_path
                )

            else:

                print()
                print(
                    "Best model not improved."
                )

            self.save_history()

        print()
        print("=" * 60)
        print("Training complete.")
        print("=" * 60)

        print(
            f"Best mean tumor Dice: "
            f"{self.best_mean_tumor_dice:.4f}"
        )

        return self.history
