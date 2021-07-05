# cv/classificationdataset.py
# Creates dataset for classification tasks

import torch
import torchvision as tv
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import PIL

from skimage import io
import numpy as np


from typing import Any, Optional, Dict, Tuple
from pathlib import Path
import random


#Define mean adn standard deviation
MEANS = np.asarray([0.485, 0.456, 0.406])
STDS = np.asarray([0.229, 0.224, 0.225])

#define transform
normalize = tv.transforms.Normalize(mean=MEANS, std=STDS, inplace=True)
traintransform = tv.transforms.Compose([
    tv.transforms.RandomRotation(degrees=(-90, 90)),
    tv.transforms.RandomHorizontalFlip(p=0.5),
    tv.transforms.RandomVerticalFlip(p=0.1),
    tv.transforms.RandomGrayscale(p=0.1),
    tv.transforms.ColorJitter(brightness=.25, contrast=.25, saturation=.25),
    tv.transforms.ToTensor(),
    normalize
])

valtransform = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    normalize
])


class ClassificationDataset(Dataset):
    def __init__(self, datadir:str, fold:str, imagesize:int, labelmapping:Dict[str, int]):
        super(ClassificationDataset, self).__init__()
        self.datadir = Path(datadir)
        self.fold = fold
        self.labelmapping = labelmapping
        
        imagedir = self.datadir/self.fold
        self.imagepaths = [str(path) for path in imagedir.glob('*/*')]
        self.len = len(self.imagepaths)
        
        random.shuffle(self.imagepaths)
        self.imagesize = imagesize
        
        
    def __getitem__(self, index:int) -> Tuple[Any, ...]:
        imagepath = self.imagepaths[index]
        imagelabel = self.labelmapping[imagepath.split('/')[-2]]
    
        
        #read image
        image = default_loader(imagepath)

        
        if self.fold == 'train':
            image =  tv.transforms.RandomResizedCrop(self.imagesize).forward(image)
            image = traintransform(image)
            
        else:
            # resizes smaller edge to img_size
            image = tv.transforms.Resize(self.imagesize, interpolation=PIL.Image.BICUBIC).forward(image)
            image = tv.transforms.CenterCrop(self.imagesize).forward(image)
            image = valtransform(image)
            
        #TODO - weights to labels            
        return image, imagelabel, imagepath
    
    def __len__(self):
        return self.len        