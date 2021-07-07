# cv/train.py
# Training operations

import warnings
warnings.filterwarnings('ignore')
import yaml
import os
from collections import OrderedDict
from typing import Sequence, Dict, Optional, Sequence, Any, Tuple
from tqdm import tqdm
import numpy as np
import torch
from cv.utils import *




# ## Train and Validate

class Trainer:
    """Object to facilitate training."""
    def __init__(
        self, 
        model: torch.nn.Module,
        config: Dict[str, Any],
        device: Optional[torch.device] = torch.device('cpu'),
        lossfn: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.nn.Module] = None,
        topk: Optional[Sequence] = (1, 3)
    ):
        self.model = model
        self.device = device
        self.lossfn = lossfn
        self.optimizer = optimizer
        self.topk = topk
        self.config = config

    def runepoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        step: str
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        assert 'pred' not in step
        print(f'Starting {step}')
        istrain = True if 'train' in step else False
        #set dropout and batchnorm to false in eval and during finetuning
        model.train(istrain)

        #define average losses and accuracy tracking objects
        losses = Averagemeter()
        topkaccuracy = {f'top{k}': Averagemeter() for k in top}

        #declare object to store targets and predictions
        predictions, targets = list(), list()

        #set tqdm iterator
        iterator = tqdm(dataloader)
        description = list() #description that is used to show in tqdm 

        with torch.set_grad_enabled(istrain):
            #loop through each batch of data
            for batch in iterator:
                batch = [item.to(self.device) for item in batch]
                images, labels, _ = batch
                batchsize = self.images.size(0)

                #predict the labels
                outputs = self.model(images)


                #calculate loss
                loss = self.lossfn(outputs, labels)
                loss = loss.mean()
                losses.update(loss.item(), batchsize)
                #update the description with current loss and avg loss
                description.append(f'Loss: {losses.val:.3f}|({losses.avg:.3f})')

                #collect data
                targets.extend(labels.cpu().numpy())
                predictions.extend(torch.sigmoid(outputs).cpu().numpy())

                #if training do backward pass
                if istrain:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                #calculate top accuracy
                accuracy = self.accuracy(outputs, labels, self.topk)
                for k, acc in accuracy.items():
                    topkaccuracy[f'top{k}'].update(acc * 100. / batchsize, n=batchsize)
                    description.append(f'Accuracy{k}: \
                            {topkaccuracy[f"top{k}"].val:.3f}|\
                            ({topkaccuracy[f"top{k}"].avg:.3f})')

                tqdm.set_description(' '.join(description))
    
        metrics = {}
        metrics['loss'] = losses.avg
        for k, acc in topkaccuracy.items():
            metrics[f'top{k}'] = acc.avg

        return metrics, np.vstack(targets), np.vstack(predictions)

    
            
    def accuracy(
        self, 
        outputs: torch.Tensor, 
        labels: torch.Tensor,
        top: Sequence[int] = (1,)
    ) -> Dict[int, float]:
        with torch.no_grad():
            # preds and labels both have shape [N, k]
            _, preds = outputs.topk(k=max(top), dim=1, largest=True, sorted=True)
            labels = labels.view(-1, 1).expand_as(preds)

            corrects = preds.eq(labels).cumsum(dim=1) 
            corrects = corrects.sum(dim=0)  # shape [k]     
            tops = {k: corrects[k - 1].item() for k in top}
            
        return tops
