from __future__ import annotations

from abc import ABC, abstractmethod
import glob
import os
from random import random

import torch
from torch import Tensor
from numpy import ndarray
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import functional as F

class Augmentation(ABC):
    @abstractmethod
    def __init__(self):
        ...
    
    @abstractmethod
    def __call__(self, image, target) -> tuple:
        ...


class Compose:
    def __init__(self, tfms: list[Augmentation]):
        """
        Compose sequential augmentations.

        Parameters
        ----------
        tfms: list[Augmentation]
            List of transforms following inhereted from Augmentation.
        """
        self.tfms = [tfm for tfm in tfms if issubclass(tfm, Augmentation)]

    def __call__(
        self, image: ndarray | Image, target: ndarray | Image
    ) -> tuple[Tensor, Tensor]:
        """
        Call Compose.

        Parameters
        ----------
        image : ndarray | Image | Tensor
            Image to augment.
        target : ndarray | Image | Tensor
            Corresponding mask.

        Returns
        -------
        tuple[Tensor]
            Augmented image and corresponding mask.
        """
        for tfm in self.tfms:
            image, target = tfm(image, target)

        return image, target


class ToTensor(Augmentation):
    def __init__(self):
        """
        Convert input to torch.Tensor
        """
        super().__init__()

    def __call__(
        self, image: ndarray | Image, target: ndarray
    ) -> tuple[Tensor, Tensor]:
        """
        Call ToTensor.

        Parameters
        ----------
        image : ndarray | Image
            Image to convert to Tensor.
        target : ndarray
            Mask to convert to Tensor.
        """
        if isinstance(image, (ndarray, Image)):
            image = F.to_tensor(image)

        if isinstance(target, ndarray):
            target = torch.from_numpy(target)

        return image, target


class ConvertImageDtype*(Augmentation):
    def __init__(self, dtype: torch.dtype):
        """
        Convert input Tensor to a different data type.

        Parameters
        ----------
        dtype : torch.dtype
            PyTorch data type to convert inputs to.
        """
        self.dtype = dtype

    def __call__(self, image: Tensor, target: Tensor) -> tuple[Tensor]:
        """
        Call ConvertImageDtype.

        Parameters
        ----------
        image : Tensor
            Image to convert.
        target : Tensor
            Corresponding mask.

        Returns
        -------
        tuple[Tensor]
            Converted image and original mask.
        """
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


class RandomHorizontalFlip(Augmentation):
    def __init__(self, flip_prob: float = 0.5):
        """
        Randomly flip image and target horizontally.

        Parameters
        ----------
        flip_prob : float
            Probability a flip will occur. Must be in between 0 and 1, defaults to 0.5.
        """
        self.flip_prob = flip_prob

    def __call__(
        self, image: Tensor | Image, target: Tensor | Image
    ) -> tuple[Tensor]:
        """
        Call RandomHorizontalFlip.

        Parameters
        ----------
        image : Tensor | Image
            Image to flip.
        target : Tensor | Image
            Mask to flip.

        Returns
        -------
        tuple[Tensor]
            Flipped (or not) image and target.
        """
        if random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class Normalize(Augmentation):
    def __init__(self, mean: list[float], std: list[float]):
        """
        Normalize image with mean and standard deviation for each channel.

        Parameters
        ----------
        mean : list[float]
            Mean for every channels.
        std : list[float]
            Standard deviation for every channel.
        """
        self.mean = mean
        self.std = std

    def __call__(self, image: Tensor, target: Tensor) -> tuple[Tensor]:
        """
        call Normalize.

        Parameters
        ----------
        image : Tensor
            Image to normalize.
        target : Tensor
            Corresponding mask.
        
        Returns
        -------
        tuple[Tensor]
            Normalized image and original target.
        """
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

    @classmethod
    def from_string(cls, string: str) -> Normalize:
        """
        Initialize class from string.

        Parameters
        ----------
        string : str
            Name of dataset to use.
        
        Returns
        -------
        Normalize
            Instance with mean and standard deviations for the dataset.
        """
        if string.lower() == "imagenet":
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            raise ValueError(f"{string} is not currently supported")
        
        return cls(mean, std)


class Scale(Augmentation):
    def __init__(self, new_min: int = 0, new_max: int = 1):
        """
        Scale image to new minimum and maximum.

        Parameters
        ----------
        new_min : int
            Minimum of scaled data.
        new_max : int
            Maximum of scaled data.
        """
        self.new_min = new_min
        self.new_max = new_max

    def __call__(self, image: Tensor, target: Tensor) -> tuple[Tensor]:
        """
        Call Scale.

        Parameters
        ----------
        image : Tensor
            Image to scale.
        target : Tensor
            Corresponding mask.
        
        Returns
        -------
        tuple[Tensor]
            Scaled image and original target.
        """
        image = (image - image.min()) / (image.max() - image.min()) * (
            self.new_max - self.new_min
        ) + self.new_min

        return image, target
