# cv/utils.py
# Utils to support training

import torch
from typing import Mapping, Dict, Any

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
        
        
def prefix_all_keys(d: Mapping[str, Any], prefix: str) -> Dict[str, Any]:
    """Returns a new dict where the keys are prefixed by <prefix>."""
    return {f'{prefix}{k}': v for k, v in d.items()}