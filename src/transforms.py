from __future__ import annotations

from numpy import ndarray
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

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
