#!/usr/bin/env python

import torch
from torch.nn import functional as F
import pytorch_lightning as pl
import hydra
from hydra.utils import instantiate
from pathlib import Path
from torchvision.models import resnet18
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os

debug = False

class BagDataModule(pl.LightningDataModule):
    def __init__(self, root, batch_size):
        super().__init__()
        self.batch_size = batch_size
        data_transform = transforms.Compose([
           transforms.RandomHorizontalFlip(),
           transforms.ToTensor()
        ])
        imageDir = ImageFolder(root, transform=data_transform)
        train_size = int(.8 * len(imageDir)) 
        test_size = len(imageDir) - train_size # remainder 
        train,test = random_split(imageDir,[train_size,test_size])

        self.train_dataset = train
        self.val_dataset = test

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)


class BagClassifier(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # init a pretrained resnet
        backbone = resnet18(pretrained=True)
        layers = list(backbone.children())[:-1] # remove last layer
        self.base_model = torch.nn.Sequential(*layers)

        # use the pretrained model to classify backpack or briefcase (2 image classes)
        num_target_classes = 2
        self.classifier = torch.nn.Linear(backbone.fc.in_features, num_target_classes)
        self.accuracy = pl.metrics.Accuracy()

    def forward(self, x):
            self.base_model.eval()
            with torch.no_grad():
                features = self.base_model(x).flatten(1)
            return self.classifier(features)
            
    def training_step(self, batch, batch_idx, key="train"):
        im, label = batch
        out = self.forward(im)
        loss = F.cross_entropy(out, label)
        pred = out.argmax(dim=-1)

        self.log(f"{key}_loss", loss, prog_bar=True)
        self.log(f"{key}_acc", self.accuracy(pred, label), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx, "val")
        return loss

    def configure_optimizers(self):
        return Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.learning_rate)


@hydra.main(config_path="config", config_name="base.yaml")
def main(config):
    trainer = pl.Trainer(**config.trainer, fast_dev_run=debug)
    trainer.fit(instantiate(config.module), instantiate(config.data_module))

if __name__ == "__main__":
    main()

    