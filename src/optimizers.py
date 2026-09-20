import torch
from torch.optim import Optimizer

from SpikeKFAC.kfac_stab import SpikeFAC as SpikeFAC_Stab


def get_optimizer(optimizer_name, opt):
    if optimizer_name == 'adam':
        return torch.optim.Adam
    elif optimizer_name == 'adamw':
        return torch.optim.AdamW
    elif optimizer_name == 'sgd':
        return torch.optim.SGD
    elif optimizer_name == 'spikefac_stab':
        return SpikeFAC_Stab
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
