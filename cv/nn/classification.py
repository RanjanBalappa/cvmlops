import torch
import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet



class Classification(nn.Module):
    def __init__(self, extractor:str, numclasses:int):
        super(Classification, self).__init__()
        print(extractor)
        self.encoders = {
        'resnet101': models.resnet101(pretrained=True),
        'resnet50': models.resnet50(pretrained=True),
        'efficientnet': EfficientNet.from_pretrained(extractor if 'efficient' in extractor else 'efficientnet-b0', 
                                                        num_classes=numclasses),
        'inceptionv3': models.inception_v3(pretrained=True)

        }
        self.encoder = self.encoders[extractor.split('-')[0]]
        if 'inception' in extractor:
            numftrs = self.encoder.AuxLogits.fc.in_features
            self.encoder.AuxLogits.fc = nn.Linear(numftrs, numclasses)
            
        if 'efficientnet' not in extractor:
            self.encoder.fc = nn.Linear(self.encoder.fc.in_features, numclasses, bias=True)

        else:
            self.encoder._fc = nn.Linear(self.encoder._fc.in_features, numclasses, bias=True)
         

    def forward(self, images):
        return self.encoder(images)
