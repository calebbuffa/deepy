from __future__ import annotations

import glob
import os
from random import random

import torch
from torch import Tensor
from numpy import ndarray
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import functional as F


class Compose:
    def __init__(self, tfms: list):
        self.tfms = tfms

    def __call__(
        self, image: ndarray | Image, target: ndarray | Image
    ) -> tuple[Tensor, Tensor]:
        for tfm in self.tfms:
            image, target = tfm(image, target)

        return image, target


class ToTensor:
    def __init__(self):
        super().__init__()

    def __call__(
        self, image: ndarray | Image, target: ndarray
    ) -> tuple[Tensor, Tensor]:
        if isinstance(image, (ndarray, Image)):
            image = F.to_tensor(image)

        if isinstance(target, ndarray):
            target = torch.from_numpy(target)

        return image, target


class ConvertImageDtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image: Tensor, target: Tensor) -> tuple[Tensor]:
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, flip_prob: float):
        self.flip_prob = flip_prob

    def __call__(
        self, image: Tensor | Image, target: Tensor | Image
    ) -> tuple[Tensor, Tensor]:
        if random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class Normalize:
    def __init__(self, mean: list[float], std: list[float]):
        self.mean = mean
        self.std = std

    def __call__(self, image: Tensor, target: Tensor) -> tuple[Tensor]:
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class Scale:
    def __init__(self, new_min: int, new_max: int):
        self.new_min = new_min
        self.new_max = new_max

    def __call__(self, image: Tensor, target: Tensor) -> tuple[Tensor]:
        image = (image - image.min()) / (image.max() - image.min()) * (
            self.new_max - self.new_min
        ) + self.new_min

        return image, target
