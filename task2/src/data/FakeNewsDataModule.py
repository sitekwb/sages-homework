from typing import Optional
from data.FakeNewsDataset import FakeNewsDataset
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
import torch

from data.make_dataset import make_dataset


class FakeNewsDataModule(pl.LightningDataModule):
    def __init__(self, bodies_path: str, stances_path: str, w2v_model_path: str, text_input_len: int,
                 batch_size: int = 32, train_frac: float = 0.7, test_frac: float = 0.2, num_workers=0):
        super().__init__()
        self.bodies_path: str = bodies_path
        self.stances_path: str = stances_path
        self.w2v_model_path: str = w2v_model_path
        self.batch_size: int = batch_size
        self.train_frac: float = train_frac
        self.test_frac: float = test_frac
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.num_workers = num_workers
        self.dataset_size = None
        self.text_input_len = text_input_len

    def prepare_data(self) -> None:
        self.dataset_size = make_dataset(self.bodies_path, self.stances_path, self.w2v_model_path)

    def setup(self, stage: Optional[str] = None):
        train_size: int = int(self.train_frac * self.dataset_size)
        test_size: int = int(self.test_frac * self.dataset_size)
        val_size: int = self.dataset_size - train_size - test_size
        self.dataset = FakeNewsDataset(bodies_path=self.bodies_path,
                                       stances_path=self.stances_path,
                                       w2v_model_path=self.w2v_model_path,
                                       text_input_len=self.text_input_len)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset,
                                                                               [train_size, val_size, test_size],
                                                                               generator=torch.Generator().manual_seed(
                                                                                   42))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
