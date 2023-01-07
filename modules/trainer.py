import numpy as np
from typing import Optional
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from .config import Config

from tqdm import tqdm
import sys
import os


class TrainerOutput:
    loss: float

    def __init__(self, loss: float):
        self.loss = loss

    @property
    def score(self) -> float:
        return self.loss


class Trainer:
    model: nn.Module
    config: Config
    optimizer: optim.Optimizer
    lr_scheduler: optim.lr_scheduler._LRScheduler
    loss_function: nn.CrossEntropyLoss
    dataloader: DataLoader
    scaler: Optional[GradScaler]

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler._LRScheduler,
        loss_function: nn.CrossEntropyLoss,
        dataloader: DataLoader,
        scaler: Optional[GradScaler] = None,
    ):
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function
        self.dataloader = dataloader
        self.scaler = scaler

    def advance(self, verbose: bool = False) -> TrainerOutput:
        scaler = None
        if self.scaler:
            scaler = self.scaler
        self.model = self.model.train()
        GPU_DEVICE = torch.device("cuda")
        epoch_loss = 0
        with autocast(enabled=True):
            with torch.enable_grad():
                for batch in tqdm(self.dataloader):
                    if len(batch) == 3:
                        imgs, ids, labels = batch
                    elif len(batch) == 2:
                        imgs, labels = batch
                    else:
                        continue
                    self.optimizer.zero_grad()
                    imgs = imgs.to(GPU_DEVICE)
                    labels = labels.to(GPU_DEVICE)
                    out: torch.Tensor = self.model(imgs)
                    out = out.float()
                    if verbose:
                        print("out-shape", out.shape)
                        print("out", out)                        
                    if verbose:
                        print("labels", labels)
                    loss = self.loss_function(out, labels)
                    if verbose:
                        print("loss", loss)
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    epoch_loss += loss.item()
                    if scaler is not None:
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        self.optimizer.step()
                self.lr_scheduler.step()
        loss = epoch_loss / (len(self.dataloader))
        return TrainerOutput(loss=loss)
    
    def save_model(self):
        config = self.config
        out_path = os.path.join("./", "weights")
        os.makedirs(out_path, exist_ok=True)
        out_path = os.path.join(out_path, f"epoch_{config.n_epochs}_base_{config.model_name}.pth")
        torch.save(self.model.state_dict(), out_path)
