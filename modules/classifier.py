from typing import Callable, Dict, List, Tuple, Union, Optional
from typing_extensions import Self
import numpy as np
from torch import Tensor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
import os
import pandas as pd
from tqdm import tqdm

class ClassifierOutput:
    move_to_full_nn_img_ids: List[str]
    count: int
    acc: float
    loss: float
    df: pd.DataFrame

    def __init__(
        self,
        count: int,
        acc: float,
        loss: float,
        df: pd.DataFrame = pd.DataFrame()
    ):
        self.count = count        
        self.acc = acc
        self.loss = loss
        self.df = df

    @property
    def score(self) -> float:
        return self.acc


class ClassifierTrainInput:
    def __init__(self):
        pass

class ClassifierTestInput:
    def __init__(self):
        pass

class Classifier:

    model: nn.Module
    activate_function: nn.Softmax
    loss_function: Optional[nn.CrossEntropyLoss]
    c_high: float
    c_low: float
    dataloader: DataLoader
    phase: str
    phase_input: Union[ClassifierTrainInput, ClassifierTestInput]
    on_classify: Optional[Callable]

    def __init__(
        self,
        model: nn.Module,
        activate_function: nn.Softmax,
        dataloader: DataLoader,
        phase: str,
        phase_input: Union[ClassifierTrainInput, ClassifierTestInput],
        loss_function: Optional[nn.CrossEntropyLoss] = None,
        on_classify: Optional[Callable] = None,
    ):
        self.model = model
        self.activate_function = activate_function
        self.loss_function = loss_function
        self.dataloader = dataloader
        self.phase = phase
        self.phase_input = phase_input
        self.on_classify = on_classify

    def infer(
        self, verbose: bool = False, handle_all: bool = False, calc_acc: bool = True
    ) -> ClassifierOutput:
        net = self.model
        loader = self.dataloader
        loss_function = self.loss_function
        count = 0
        correct = 0        
        GPU_DEVICE = torch.device("cuda")
        called = False
        net = net.eval()
        epoch_loss = 0
        label_df = pd.DataFrame()
        with torch.no_grad():
            for batch in tqdm(loader):
                if len(batch) == 3:
                    imgs, ids, labels = batch
                elif len(batch) == 2:
                    imgs, ids = batch
                    labels = None
                else:
                    continue
                imgs = imgs.to(GPU_DEVICE)
                if labels is not None:
                    labels = labels.to(GPU_DEVICE)                
                outputs = net(imgs)
                if loss_function and labels is not None:
                    loss = loss_function(outputs, labels)
                    epoch_loss += loss.item()
                outputs = self.activate_function(outputs)
                if verbose:
                    print("output", outputs[:10])
                _, pred_indexes = torch.max(outputs, dim=1)
                pred_indexes: Tensor = pred_indexes
                if verbose:
                    print(f"{self.phase}_predicted_indexes", pred_indexes)
                    print(f"{self.phase}_labels", labels)
                if not called and self.on_classify is not None:
                    if not handle_all:
                        called = True
                    self.on_classify(imgs, pred_indexes)
                
                if labels is not None:
                    labels = torch.tensor(list(map(lambda nums: (nums==1).nonzero().item(), labels))).to(GPU_DEVICE)
                count += pred_indexes.shape[0]
                batch_df = pd.DataFrame()
                batch_df["image_id"] = ids
                batch_df["label"] = pd.Series(pred_indexes.detach().cpu().numpy())
                if len(label_df) == 0:
                    label_df = batch_df
                else:
                    label_df = pd.concat([label_df, batch_df])              
                if labels is not None and calc_acc:
                    correct += int(torch.where(pred_indexes == labels, 1.0, 0.0).sum())                    
            acc = correct / count
            epoch_loss = epoch_loss / (len(loader))
            label_df.reset_index(drop=True, inplace=True)
            ret = ClassifierOutput(
                count=count,
                acc=acc,
                loss=epoch_loss,
                df=label_df
            )            
            return ret
