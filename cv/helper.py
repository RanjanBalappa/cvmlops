from typing import Dict, Optional
from argparse import Namespace
from cv.nn.classification import Classification
import torch
from torch.optim import Adam, SGD, RMSprop



class Helper:
    def __init__(
        self, 
        params: Namespace
    ):
        self.params = params
        task = self.params.task
        self.model = task

    @property
    def model(self):
        return self._model


    @model.setter
    def model(self, task) -> torch.nn.Module:
        config = self.params.tasks[task]
        if task == 'classification':
            _model = Classification(extractor=config['extractor'], numclasses=config['numclasses'])

        #Load any pretrained weights provided in params
        startepoch = -1
        if self.params.pretrained  != 'imagenet' and self.params.pretrainedpath !='':
            print(f"Loading the checkpoint from {self.params.pretrainedpath}")
            checkpoint = torch.load(self.params.pretrainedpath, map_location='cpu')
            _model.load_state_dict(checkpoint)

        #load the model to device
        _model.to(self.device)
        if self.params.finetune:
            _model.requires_grad_(False)
            for param in _model.encoder.fc.parameters():
                param.requires_grad = True

        else:
            _model.requires_grad_(True)


        self._model = _model


    @property
    def device(self) -> torch.device:
        # ## Set Device
        cudaavailable = torch.cuda.is_available()
        _device = torch.device('cuda') if cudaavailable else torch.device('cpu')
        return _device


    @property
    def lossfn(self) -> torch.nn.Module:
        if self.params.lossfn == 'crossentropyloss':
            _lossfn = torch.nn.CrossEntropyLoss()

        _lossfn.to(self.device)
        return _lossfn


    @property
    def optimizer(self) -> torch.nn.Module:
        name = self.params.optimizer['name'].lower()
        if name == 'sgd':
            _optimizer = SGD(
                            self.model.parameters(), 
                            lr=float(self.params.optimizer['initlr']), 
                            momentum=float(self.params.optimizer['momentum']), 
                            weight_decay=float(self.params.optimizer['momentum'])
                        )

        elif name == 'rmsprop':
            _optimizer = RMSprop(
                                self.model.parameters(), 
                                lr=float(self.params.optimizer['initlr']), 
                                alpha=float(self.params.optimizer['alpha']),
                                momentum=float(self.params.optimizer['momentum']), 
                                weight_decay=float(self.params.optimizer['weightdecay'])
                        )
        
        elif name == 'adam':
            _optimizer = Adam(
                            self.model.parameters(), 
                            lr=float(self.params.optimizer['initlr'])
                        )

        return _optimizer




    
