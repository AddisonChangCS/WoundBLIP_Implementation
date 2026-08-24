import random
import os,torch

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import Optimizer, AdamW
from omegaconf import DictConfig
"""Fine-tune stage — training loop.
 
fit() is the sole entry point, called from main.py.
"""
class WoundBLIPTrainer():
    def __init__(
            self,
            cfg: DictConfig, 
            train_loader: DataLoader, 
            val_loader: DataLoader,
            model,
            optimizer: Optimizer | None = None
    ):
        self.cfg=cfg
        self.train_loader=train_loader
        self.val_loader=val_loader
        self.model=model

        if optimizer is None:
            self.optimizer = AdamW(self.model.parameters(), lr=cfg.training.lr)
        else:
            self.optimizer = optimizer

        self._train_losses = []
        self._val_losses = []
        self._best_val_loss = float('inf')

        self.run_dir = Path(self.cfg.run_dir)
        self.best_model_path = self.run_dir / "model"

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
#--------------------------------getter------------------------------------#
    @property
    def train_losses(self):
        return self._train_losses
    @property
    def val_losses(self):
        return self._val_losses
    @property
    def best_val_loss(self):
        return self._best_val_loss
#--------------------------------utiles------------------------------------#
    @staticmethod
    def seed_everything(seed=42):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
        print(f"Global seed set to {seed}")

    @staticmethod
    def plot_loss_curve(
        train_losses: list[float],
        vali_losses: list[float],
        run_dir: Path,
    ) -> None:
        epochs = np.arange(1, len(train_losses) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses, color="#2563eb", linewidth=2,
             marker="o", markersize=5, label="Train Loss")
        plt.plot(epochs, vali_losses, color="#f97316", linewidth=2,
             marker="o", markersize=5, label="Validation Loss")
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Training vs Validation Loss", fontsize=14, fontweight="bold")
        plt.xticks(epochs)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.legend(fontsize=11, frameon=False)
        plt.annotate(f"{train_losses[-1]:.4f}", xy=(epochs[-1], train_losses[-1]),
                    xytext=(5, -5), textcoords="offset points", fontsize=9, color="#2563eb")
        plt.annotate(f"{vali_losses[-1]:.4f}", xy=(epochs[-1], vali_losses[-1]),
                    xytext=(5, 5), textcoords="offset points", fontsize=9, color="#f97316")
        plt.tight_layout()
        plt.savefig(run_dir / "loss_curve.png", dpi=150)
        plt.close()
        print(f"Loss curve saved to {run_dir / 'loss_curve.png'}")
#-----------------------------core----------------------------------------#
    def _run_epoch(self, dataloader: DataLoader, train: bool = True) -> float:
        self.model.train() if train else self.model.eval()
        total_loss = 0.0
 
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for batch in dataloader:
                pixel_values = batch["pixel_values"].to(self.cfg.device)
                input_ids = batch["input_ids"].to(self.cfg.device)
                attention_mask = batch["attention_mask"].to(self.cfg.device)
                labels=batch["labels"].to(self.cfg.device)
 
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
 
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.training.get("max_norm", 1.0))
                    self.optimizer.step()
                total_loss += loss.item()
 
        return total_loss / len(dataloader)

#------------------------entry point----------------------------#
    def train(self) -> Path:
        """Fine-tune BLIP and save best model + test split for inference.

        Returns:
            Path: run output directory
        """
        # training loop
        for epoch in range(self.cfg.training.epochs):
            t_loss = self._run_epoch(
                dataloader=self.train_loader, 
                train=True,
            )
            v_loss = self._run_epoch(
                dataloader=self.val_loader,  
                train=False,
            )

            self._train_losses.append(t_loss)
            self._val_losses.append(v_loss)
            print(f"Epoch {epoch+1}/{self.cfg.training.epochs} | "
                f"train={t_loss:.4f} | vali={v_loss:.4f}")
 
            if v_loss < self._best_val_loss:
                self._best_val_loss = v_loss
                self.model.save_pretrained(self.best_model_path)
                print("\033[32m  -> saved best model\033[0m")
 
        self.plot_loss_curve(self.train_losses, self.val_losses, self.run_dir)
 
        return self.best_model_path