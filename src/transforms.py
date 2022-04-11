from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
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
    
class Resize(Augmentation):
    train_only: bool = False

    def __init__(self, size: list[int]):
        self.size = size

    def __call__(self, image: Tensor, target: Optional[Tensor] = None) -> tuple[Tensor]:
        image = F.resize(image, self.size)
        if target is not None:
            target = F.resize(target, self.size)
            return image, target
        return image

class Affine(Augmentation):
    def __init__(
        self,
        angle: float = -90.0,
        translate: tuple[int] = (0.1, 0.3),
        scale: float = 1.0,
        shear: float = 90.0,
    ):
        self.angle = angle
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def __call__(
        self, image: Tensor | Image, target: Optional[Tensor | Image] = None
    ) -> tuple[Tensor | Image, Optional[Tensor | Image]]:
        if isinstance(image, (Tensor, Image)):
            image = F.affine(image, self.angle, self.translate, self.scale, self.shear)

        if isinstance(target, (Tensor, Image)):
            target = F.affine(
                target, self.angle, self.translate, self.scale, self.shear
            )

        return image, target
    
 class RandomVerticalFlip(Augmentation):

    def __init__(self, flip_prob: float = 0.5):
        self.flip_prob = flip_prob

    def __call__(
        self, image: Tensor | Image, target: Optional[Tensor | Image] = None
    ) -> tuple[Tensor, Tensor]:
        if random() < self.flip_prob:
            image = F.vflip(image) if isinstance(image, (Tensor, Image)) else image
            target = F.vflip(target) if isinstance(target, (Tensor, Image)) else target
        return image, target
    
class RandomInvert(Augmentation):
    train_only: bool = True

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self, image: Tensor | Image, target: Optional[Tensor | Image] = None
    ) -> tuple[Tensor | Image, Optional[Tensor | Image]]:
        if random() < self.p:
            image = F.invert(image) if isinstance(image, (Tensor, Image)) else image
        return image, target
    

class RandomSolarize(Augmentation):

    def __init__(self, threshold: float):
        self.threshold = threshold

    def __call__(
        self, image: Tensor | Image, target: Tensor
    ) -> tuple[Tensor, Optional[Image | Tensor]]:
        image = F.solarize(image, self.threshold)
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


class ConvertImageDtype(Augmentation):
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

class GaussianBlur(Augmentation):

    def __init__(self, kernel_size: int, sigma: tuple[float] = (0.1, 2.0)):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(
        self, image: Tensor | Image, target: Optional[Tensor | Image] = None
    ) -> tuple[Tensor | Image, Optional[Tensor | Image]]:
        if isinstance(image, (Tensor, Image)):
            image = F.gaussian_blur(image, self.kernel_size, self.sigma)
        return image, target

class Grayscale(Augmentation):

    def __init__(self, num_output_channels: int = 1):
        self.num_output_channels = num_output_channels

    def __call__(
        self, image: Image, target: Optional[Tensor | Image] = None
    ) -> tuple[Tensor | Image, Optional[Tensor | Image]]:
        if isinstance(image, Image):
            image = F.to_grayscale(image, num_output_channels=self.num_output_channels)
        return image, target
    
class ColorJitter(Augmentation):

    def __init__(
        self,
        brightness: float = 1.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        hue: float = 1.0,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(
        self, image: Tensor | Image, target: Optional[Tensor | Image] = None
    ) -> tuple[Tensor | Image, Optional[Tensor | Image]]:
        if isinstance(image, Tensor, Image):
            image = F.adjust_brightness(image, self.brightness)
            image = F.adjust_contrast(image, self.contrast)
            image = F.adjust_saturation(image, self.saturation)
            image = F.adjust_hue(image, self.hue)
        return image, target
    
class RandomCrop(Augmentation):

    def __init__(self, size: int):
        self.size = size

    def __call__(
        self, image: Tensor | Image, target: Optional[Tensor | Image] = None
    ) -> tuple[Tensor | Image, Optional[Tensor | Image]]:
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        if isinstance(image, (Tensor, Image)):
            image = F.crop(image, *crop_params)

        if isinstance(target, (Tensor, Image)):
            target = F.crop(target, *crop_params)

        return image, target
