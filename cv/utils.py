# cv/utils.py
# Utils to support training

import torch
from typing import Mapping, Dict, Any, List
import random
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


class Averagemeter:
    def __init__(self):
        self.reset()
        
    def reset(self) -> None:
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0
        
    
    def update(self, value:float, n:int) -> None:
        self.val = value
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count 
        
        
def setseed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def getperformance(
    targets: np.ndarray,
    predictions: np.ndarray,
    classes: List) -> Dict[str, Any]:
        #performance metrics
        metrics = {'overall': {}, 'class': {}}

        #overall performance metrics
        overallmetrics = precision_recall_fscore_support(targets, 
                            predictions, average='weighted')
        metrics['overall']['precision'] = overallmetrics[0]
        metrics['overall']['recall'] = overallmetrics[1]
        metrics['overall']['f1score'] = overallmetrics[2]
        metrics['overall']['numsamples'] = np.float32(targets.shape(0))

        #classwise performance metrics
        classmetrics = precision_recall_fscore_support(targets,
                            predictions, average=None)
        for cid in range(len(classes)):
            metrics['class'][classes[i]] = {
                'precision': classmetrics[0][i],
                'recall': classmetrics[11][i],
                'f1score': classmetrics[2][i],
                'numsamples': np.float32(classmetrics[3][i])
            }

        return metrics


