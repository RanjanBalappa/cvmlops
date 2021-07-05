# cv/train.py
# Training operations

import warnings
warnings.filterwarnings('ignore')
import yaml
import os
from collections import OrderedDict
from typing import Sequence, Dict, Optional, Sequence
from tqdm import tqdm
import torch
from torch.optim import Adam, SGD, RMSprop
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils import tensorboard

from nn.classification import Classification
from classificationdataset import ClassificationDataset
from utils import *


# ## Load Task Config
configfile = open('config.yml')
config = yaml.load(configfile, Loader=yaml.FullLoader)
task = config['task']
taskconfig = config['tasks'][task.lower()]
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

cudaavailable = torch.cuda.is_available()
device = torch.device('cuda') if cudaavailable else torch.device('cpu')
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


# ## Create Tensorboard writer to write logs
logdir = config['logdir']
writer = tensorboard.SummaryWriter(config['logdir'])


# ## Train and Validate
def setfinetune(model: torch.nn.Module, finetune: bool) -> None:
    extractor = taskconfig['extractor']
    if finetune:
        finallayer = model.encoder._fc if 'efficientnet' in extractor else model.encoder.fc
        assert isinstance(finallayer, torch.nn.Module)

        #first set required grad to false
        model.requires_grad_(False)
        for param in finallayer.parameters():
            param.requires_grad = True
            
    else:
        model.encoder.requires_grad_(True)
    
def accuracy(outputs: torch.Tensor, labels: torch.Tensor,
            top: Sequence[int] = (1,)) -> Dict[int, float]:
    with torch.no_grad():
        # preds and labels both have shape [N, k]
        _, preds = outputs.topk(k=max(top), dim=1, largest=True, sorted=True)
        labels = labels.view(-1, 1).expand_as(preds)

        corrects = preds.eq(labels).cumsum(dim=1) 
        corrects = corrects.sum(dim=0)  # shape [k]
        
            
        tops = {k: corrects[k - 1].item() for k in top}
        
    return tops

def runepoch(model:torch.nn.Module, 
             dataloader: torch.utils.data.DataLoader,
             finetune: bool,
             device: torch.device,
             lossfn: Optional[torch.nn.Module] = None,
             optimizer: Optional[torch.nn.Module] = None,
             top: Sequence[int] = (1,)) -> Dict[str, float]:
    
    #set to eval when optimizer is none and also when finetune is True
    model.train(optimizer is not None and not finetune)
    
    #define object to track loss
    losses = Averagemeter()
    accuraciestopk = {k: Averagemeter() for k in top}
    
    tqdmloader = tqdm(dataloader)
    with torch.set_grad_enabled(optimizer is not None):
        for batch in tqdmloader:
            images, labels, imagepath = batch

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batchsize = labels.size(0)

            description = []
            outputs = model(images)

            if lossfn is not None:
                loss = lossfn(outputs, labels)
                loss = loss.mean()
                losses.update(loss.item(), batchsize)
                description.append(f'Loss {losses.val:.4f} ({losses.avg:.4f})')

            #backward pass
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            topcorrect = accuracy(outputs, labels, top=top)
            for k, acc in accuraciestopk.items():
                acc.update(topcorrect[k] * (100. / batchsize), n=batchsize)
                description.append(f'accuracy@{k} {acc.val:.3f} ({acc.avg:.3f})')

            tqdmloader.set_description(' '.join(description))
            
    metrics = {}
    metrics['loss'] = losses.avg
    for k, acc in accuraciestopk.items():
        metrics[f'accuracytop{k}'] = acc.avg
        
    return metrics

bestmetrics: Dict[str, float] = {}
top = (1, 3)
earlystopping = 0

for epoch in range(int(config['numepochs']))[startepoch + 1:]:
    print(f'Epoch: {epoch + 1}')
    writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
    
    #only finetune the final layers till the epochs less than finetuneepochs later tune entire model
    finetuneepochs = config['finetuneepochs']
    finetune = finetuneepochs > epoch
    setfinetune(model, finetune)
    
    
    print('Training')
    trainmetrics = runepoch(model, traindataloader, finetune, device, lossfn, optimizer, top)
    trainmetrics = prefix_all_keys(trainmetrics, prefix='train/')
    
    
    print('Validation')
    valmetrics = runepoch(model, valdataloader, finetune, device, lossfn, top=top)
    valmetrics = prefix_all_keys(valmetrics, prefix='val/')
    
    
    scheduler.step()
    
    if valmetrics['val/accuracytop1'] > bestmetrics.get('val/accuracytop1', 0):
        filename = os.path.join(logdir, 'ckpt', f'ckpt_{epoch}.pt')
        print(f'New best model! Saving checkpoint to {filename} with accuracy {valmetrics["val/accuracytop1"]}')
        state = {
            'epoch': epoch,
            'model': getattr(model, 'module', model).state_dict(),
            'val/acc': valmetrics['val/accuracytop1'],
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, filename)
        bestmetrics.update(trainmetrics)
        bestmetrics.update(valmetrics)
        bestmetrics['epoch'] = epoch  
        earlystopping = 0
        
    else:
        earlystopping += 1
        
    if earlystopping > 9:
        print(f'Model did not improve for 10 epochs so exiting!')
