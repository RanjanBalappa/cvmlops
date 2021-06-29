import torch
import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet



class Classification(nn.Module):
    def __init__(self, extractor:str, numclasses:int):
        super(Classification, self).__init__()
        self.encoders = {
        'resnet101': models.resnet101(pretrained=True),
        'resnet50': models.resnet50(pretrained=True),
        'efficientnet': EfficientNet.from_pretrained(extractor),
        'inceptionv3': models.inception_v3(pretrained=True)

        }
        self.encoder = self.encoders[extractor.split('-')[0]]
        if 'inception' in extractor:
            numftrs = self.encoder.AuxLogits.fc.in_features
            self.encoder.AuxLogits.fc = nn.Linear(numftrs, numclasses)
        
        self.encoder.fc = self.encoder._fc if 'efficientnet' in extractor else self.encoder.fc 
        self.encoder.fc = nn.Sequential(
            nn.Linear(self.encoder.fc.in_features, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(32, numclasses, bias=True)
        )

    def forward(self, images):
        return self.encoder(images)
