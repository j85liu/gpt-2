from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import function as F


# --------------------------------------------------------------------------------


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layers: int = 6
    n_head: int = 6
    nembd: int = 384

class GPT(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.config = config
        
        self.transofrmer = nn.ModuleDict(dict){
            
        }