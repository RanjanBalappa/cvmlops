# cv/main.py
# Training and Optimization

import warnings
warnings.filterwarnings('ignore')
import yaml
import os
from argparse import Namespace
from collections import OrderedDict
from typing import Sequence, Dict, Optional, Sequence, Any
from tqdm import tqdm
import torch
from torch.optim import Adam, SGD, RMSprop
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, SubsetRandomSampler
from cv.nn.classification import Classification
from cv.classificationdataset import ClassificationDataset
from cv import train, utils 

#set manual seed for reproducibility
utils.setseed(1234)

def trainmodel(params: Namespace):

    # ## Load Task Config
    task = params['task']
    taskconfig = params['tasks'][task.lower()]
    print(f'The task defined is {task} with config {config}')


    # ## Load Model
    models = {
        'classification': Classification(extractor=taskconfig['extractor'], numclasses=taskconfig['numclasses'])
    }
    model = models[task]
    startepoch = -1
    if config['pretrained']  != 'imagenet' and config['pretrainedpath'] !='':
        print(f"Loading the checkpoint from {config['pretrainedpath']}")
        checkpoint = torch.load(config['pretrainedpath'], map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        startepoch = checkpoint['epoch']


    # ## Set Device
    cudaavailable = torch.cuda.is_available()
    device = torch.device('cuda') if cudaavailable else torch.device('cpu')
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    if device.type == 'cuda':
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    model.to(device)


    # ## Define loss function
    lossfns = {
        'crossentropyloss': torch.nn.CrossEntropyLoss(reduction='none').to(device)
    }
    lossfn = lossfns[config['lossfn']]


    # ## Define Optimizer
    optimizers = {
        'sgd': SGD(model.parameters(), lr=float(config['optimizer']['initlr']), 
                momentum=float(config['optimizer']['momentum']), 
                weight_decay=float(config['optimizer']['momentum'])),
        'rmsprop': RMSprop(model.parameters(), lr=float(config['optimizer']['initlr']), 
                        alpha=float(config['optimizer']['alpha']),
                        momentum=float(config['optimizer']['momentum']), 
                        weight_decay=float(config['optimizer']['weightdecay']))
    }
    optimizer = optimizers[config['optimizer']['name']]
    scheduler = lr_scheduler.StepLR(
                optimizer=optimizer, step_size=1, gamma=0.97 ** (1 / 2.4))


    # ## Load Data
    traindataset = ClassificationDataset(datadir=config['datadir'], fold='train', 
                                        imagesize=config['imagesize'], 
                                        labelmapping=taskconfig['labelmapping'])
    traindataloader = DataLoader(traindataset, 
                                shuffle=True, 
                                pin_memory=True, 
                                num_workers=6, 
                                batch_size=config['batchsize'])
    valdataset = ClassificationDataset(datadir=config['datadir'], fold='valid', 
                                    imagesize=config['imagesize'], 
                                    labelmapping=taskconfig['labelmapping'])
    valdataloader = DataLoader(valdataset, 
                                shuffle=False, 
                                pin_memory=True, 
                                num_workers=6, 
                                batch_size=config['batchsize'])



    # ## Training loop
    bestaccuracy: float = 0.0
    bestmodel: torch.nn.Module = model
    top = (1, 3)
    patience = params['earlystopping']

    trainer = train.Trainer(
        model,
        params,
        device,
        lossfn,
        optimizer
    )

    for epoch in range(int(params['numepochs']))[startepoch + 1:]:
        print(f'Epoch: {epoch + 1}')

        trainmetrics, _, _ = trainer.runepoch(traindataloader, step='train')
        valmetrics, targets, predictions = trainer.runepoch(valdataloader, step='val')
        scheduler.step()
        
        if valmetrics['top1'] > bestaccuracy:
            bestaccuracy = valmetrics['top1']
            bestmodel = model
            patience = params['earlystopping']
            
        else:
            patience -= 1
            
        if not patience:
            print(f'Model did not improve for 10 epochs so exiting!')

    #Evaluate model
    artifacts = {
        'model': bestmodel,
        'top1accuracy': bestaccuracy
    }

    #calculate performance metrics
    valmetrics, targets, predictions = trainer.runepoch(valdataloader, step='val')
    performance = utils.getperformance(targets, 
                                        predictions, 
                                        taskconfig['labelmapping'].keys()
                                        )
    artifacts['performance'] = performance
    return artifacts
   