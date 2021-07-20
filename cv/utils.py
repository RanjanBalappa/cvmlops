# cv/utils.py
# Utils to support training
import json
from typing import Mapping, Dict, Any, List
import random
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torch


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
        metrics['overall']['numsamples'] = np.float64(targets.shape[0])

        #classwise performance metrics
        classmetrics = precision_recall_fscore_support(targets,
                            predictions, average=None)
        for cid in range(len(classes)):
            metrics['class'][classes[cid]] = {
                'precision': classmetrics[0][cid],
                'recall': classmetrics[1][cid],
                'f1score': classmetrics[2][cid],
                'numsamples': np.float64(classmetrics[3][cid])
            }

        return metrics


def save_dict(d: Dict, filepath: str, cls=None, sortkeys: bool = False) -> None:
    """Save a dictionary to a specific location.
    Warning:
        This will overwrite any existing file at `filepath`.
    Args:
        d (Dict): dictionary to save.
        filepath (str): location to save the dictionary to as a JSON file.
        cls (optional): encoder to use on dict data. Defaults to None.
        sortkeys (bool, optional): sort keys in dict alphabetically. Defaults to False.
    """
    with open(filepath, "w") as fp:
        json.dump(d, indent=2, fp=fp, cls=cls, sort_keys=sortkeys)



def load_dict(filepath: str) -> Dict:
    """Load a dictionary from a JSON's filepath.
    Args:
        filepath (str): JSON's filepath.
    Returns:
        A dictionary with the data loaded.
    """
    with open(filepath) as fp:
        d = json.load(fp)
    return d