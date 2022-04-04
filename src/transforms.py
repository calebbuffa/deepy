from __future__ import annotations

from numpy import ndarray
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F

class ToTensor(transforms.ToTensor):
  def __init__(self):
      super().__init__()
  
  def __call__(self, image: ndarray | Image, target: ndarray | Image) -> tuple[Tensor, Tensor]:
    if isinstance(image, ndarray):
      image = ...
     elif isinstance(image, Image):
      image = ...
    else:
      raise TypeError
      
     if isinstance(target, ndarray):
      ...
     elif isinstance(target, Image):
      ...
     
    return image, target

  
class Compose(transforms.Compose):
  def __init__(self, tfms: list[transforms]):
    self.tfms = tfms
   
  def __call__(self, image: ndarray | Image, target: ndarray | Image) -> tuple[Tensor, Tensor]:
    for tfm in self.tfms:
      image, target = tfm(image, target)
      
  return image, target

class RandomResize(transforms.Resize):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target
      
class RandomHorizontalFlip(transforms.HorizontalFlip):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target

class Normalize(transforms.Normalize):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
