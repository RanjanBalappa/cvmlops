# cv/main.py
# Training and Optimization

import warnings
warnings.filterwarnings('ignore')
import yaml
import os
from pathlib import Path
from argparse import Namespace
from collections import OrderedDict
from typing import Sequence, Dict, Optional, Sequence, Any
from tqdm import tqdm
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, SubsetRandomSampler
import mlflow

from cv.datasets.classificationdataset import ClassificationDataset
from cv import train, utils, helper


#set manual seed for reproducibility
utils.setseed(1234)

#Data dir
BASEDIR = BASEDIR = Path(__file__).parent.parent.absolute()
DATADIR = Path(BASEDIR, 'data')

def loadartifacts(
    runid: str
) -> Dict:
    artifacturi = mlflow.get_run(run_id=runid).info.artifact_uri.split('file://')[-1]
    params = Namespace(**utils.load_dict(Path(artifacturi, 'params.json')))
    model_state = torch.load(Path(artifacturi, 'model.pt'))

    #initailize mode
    network = helper.Helper(params)
    model = network.model
    device = network.device
    lossfn = network.lossfn
    optimizer = network.optimizer

    

    #load weights
    model.load_state_dict(model_state)
    
    return {
        'params': params,
        'model': model,
        'device': device,
        'lossfn': lossfn,
        'optimizer': optimizer
    }
    

def trainmodel(params: Namespace):

    # ## Load Task Config
    task = params.task
    taskconfig = params.tasks[task.lower()]
    print(f'The task defined is {task} with config {taskconfig}')

    # ## Load Data
    traindataset = ClassificationDataset(datadir=DATADIR, fold='train', 
                                        imagesize=params.imagesize, 
                                        labelmapping=taskconfig['labelmapping'])                                  
    traindataloader = DataLoader(traindataset, 
                                shuffle=torch.utils.data.SubsetRandomSampler(range(len(traindataset))), 
                                pin_memory=True, 
                                num_workers=6, 
                                batch_size=params.batchsize)
    valdataset = ClassificationDataset(datadir=DATADIR, fold='valid', 
                                    imagesize=params.imagesize, 
                                    labelmapping=taskconfig['labelmapping'])
    valdataloader = DataLoader(valdataset, 
                                sampler=None, 
                                pin_memory=True, 
                                num_workers=6, 
                                batch_size=params.batchsize)

    #Load model, device, lossfn, optimizer
    network = helper.Helper(params)
    device = network.device
    model = network.model
    lossfn = network.lossfn
    optimizer = network.optimizer
    scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.1)


    # ## Training loop
    bestaccuracy: float = 0.0
    bestloss: float = float('inf')
    bestmodel: torch.nn.Module = model
    topk = (1,)
    patience = params.patience

    trainer = train.Trainer(
        model=model,
        config=params,
        device=device,
        lossfn=lossfn,
        optimizer=optimizer,
        topk=topk
    )

    for epoch in range(int(params.numepochs)):
        print(f'Epoch: {epoch + 1}')

        trainer.runepoch(traindataloader, step='train')
        valmetrics, targets, predictions = trainer.runepoch(valdataloader, step='val')
        scheduler.step()
        
        #check if model improved
        if valmetrics['loss'] < bestloss:
            bestaccuracy = valmetrics['top1']
            bestloss = valmetrics['loss']
            bestmodel = model
            patience = params.patience
            torch.save(model.state_dict(), f'model/{taskconfig["extractor"]}.pt')
            
        else:
            patience -= 1
            
        if not patience:
            print(f'Model did not improve for 10 epochs so exiting!')
            break

    #Evaluate model
    artifacts = {
        'model': bestmodel,
        'top1accuracy': bestaccuracy,
        'bestloss': bestloss
    }

    #calculate performance metrics
    valmetrics, targets, predictions = trainer.runepoch(valdataloader, step='val')
    performance = utils.getperformance(targets, 
                                        predictions, 
                                        list(taskconfig['labelmapping'].keys())
                                        )
    artifacts['performance'] = performance
    return artifacts


def evaluate(
    model: torch.nn.Module,
    device: torch.device,
    lossfn: torch.nn.Module,
    optimizer: torch.nn.Module,
    params: Dict,
    datafolder: str
) -> Dict[str, Any]:
    # ## task info
    task = params.task
    taskconfig = params.tasks[task.lower()]


    # #Load model, device, lossfn, optimizer
    trainer = train.Trainer(
        model=model,
        config=params,
        device=device,
        lossfn=lossfn,
        optimizer=optimizer
    )

    # ##Load dataset
    valdataset = ClassificationDataset(datadir=DATADIR, fold=datafolder, 
                                    imagesize=params.imagesize, 
                                    labelmapping=taskconfig['labelmapping'])

    valdataloader = DataLoader(valdataset, 
                                sampler=None, 
                                pin_memory=True, 
                                num_workers=6, 
                                batch_size=params.batchsize)


    _, targets, predictions = trainer.runepoch(valdataloader, step='val')
    performance = utils.getperformance(targets, 
                                        predictions, 
                                        list(taskconfig['labelmapping'].keys())
                                        )

    return performance