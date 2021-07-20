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

        #define transform
        normalize = tv.transforms.Normalize(mean=MEANS, std=STDS, inplace=True)

        if self.fold == 'train':
            self.transform = tv.transforms.Compose([
                tv.transforms.RandomResizedCrop(self.imagesize),
                tv.transforms.RandomRotation(degrees=(-90, 90)),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.RandomGrayscale(p=0.1),
                tv.transforms.ColorJitter(brightness=.25, contrast=.25, saturation=.25),
                tv.transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform = tv.transforms.Compose([
                tv.transforms.Resize(self.imagesize, interpolation=PIL.Image.BICUBIC),
                tv.transforms.CenterCrop(self.imagesize),
                tv.transforms.ToTensor(),
                normalize
            ])
        
        
    def __getitem__(self, index:int) -> Tuple[Any, ...]:
        imagepath = self.imagepaths[index]
        imagelabel = self.labelmapping[imagepath.split('/')[-2]]
    
        
        #read image
        image = default_loader(imagepath)
        image = self.transform(image)
            
        #TODO - weights to labels            
        return image, imagelabel, imagepath
    
    def __len__(self):
        return self.len        