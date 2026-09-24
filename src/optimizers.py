import torch
from torch.optim import Optimizer

from SpiKFAX.kfac_stab import SpiKFAX as SpiKFAX_Stab
from SpiKFAX.kfac import SpiKFAX as SpiKFAX_Damp

def get_optimizer(optimizer_name, opt):
    if optimizer_name == 'adam':
        return torch.optim.Adam
    elif optimizer_name == 'adamw':
        return torch.optim.AdamW
    elif optimizer_name == 'sgd':
        return torch.optim.SGD
    elif optimizer_name == 'spikfax':
        return SpiKFAX_Damp
    elif optimizer_name == 'spikfax_stab':
        return SpikeFAC_Stab
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
