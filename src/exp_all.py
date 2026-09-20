import os
import json
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import random
import numpy as np

from torch.utils.data import DataLoader
import torch
from torch.nn import CrossEntropyLoss

from datasets import get_datasets
from models import get_model
from options import Options
from trainer import Trainer
from optimizers import get_optimizer


opt = Options()
opt.dataset = 'MNIST'
opt.model = 'feedforward'
opt.optimizer = 'adam'
opt.loss = 'ce'

opt.n_epochs = 30
opt.batch_size = 128
opt.lr = 1e-4
opt.alpha = 0.95

opt.num_inputs = 28 * 28
opt.num_hidden = 256
opt.num_outputs = 10
opt.beta = 0.9
opt.n_steps = 25

def parse_args():
    parser = argparse.ArgumentParser(description='Train a spiking neural network on MNIST.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset to use (MNIST or CIFAR10).')
    parser.add_argument('--model', type=str, default='feedforward', help='Model architecture to use (feedforward or resnet).')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer to use (adam or sgd).')
    parser.add_argument('--seed', type=int, default=0, help='Seed')
    parser.add_argument('--epochs', type=int, default=30, help='Seed')
    parser.add_argument('--num_hidden', type=int, default=256, help='Seed')

    return parser.parse_args()

params = parse_args()

opt.lr = params.lr
opt.dataset = params.dataset
opt.model = params.model
opt.optimizer = params.optimizer
opt.seed = params.seed
opt.n_epochs = params.epochs
opt.num_hidden = params.num_hidden
random.seed(opt.seed)
np.random.seed(opt.seed)

if opt.dataset == 'MNIST':
    opt.num_inputs = 32 * 32
    opt.num_outputs = 10
    opt.in_channels = 1
    opt.num_classes = 10
    opt.beta = 0.9
    opt.encode = True
elif opt.dataset == 'F-MNIST':
    opt.num_inputs = 32 * 32
    opt.num_outputs = 10
    opt.in_channels = 1
    opt.num_classes = 10
    opt.beta = 0.9
    opt.encode = True
elif opt.dataset == 'CIFAR10':
    opt.num_inputs = 32 * 32 * 3
    opt.num_outputs = 10
    opt.in_channels = 3
    opt.num_classes = 10
    opt.beta = 0.9
    opt.encode = True
elif opt.dataset == 'CIFAR100':
    opt.num_inputs = 32 * 32 * 3
    opt.num_outputs = 100
    opt.in_channels = 3
    opt.num_classes = 100
    opt.beta = 0.9
    opt.encode = True
elif opt.dataset == 'N-MNIST':
    opt.num_inputs = 32 * 32 * 2
    opt.num_outputs = 10
    opt.in_channels = 2
    opt.num_classes = 10
    opt.beta = 0.9
    opt.encode = False
elif opt.dataset == 'CIFAR10-DVS':
    opt.num_inputs = 32 * 32 * 2
    opt.num_outputs = 10
    opt.in_channels = 2
    opt.num_classes = 10
    opt.beta = 0.9
    opt.encode = False
elif opt.dataset == 'DVS128-Gesture':
    opt.num_inputs = 32 * 32 * 2
    opt.num_outputs = 11
    opt.in_channels = 2
    opt.num_classes = 11
    opt.beta = 0.9
    opt.encode = False

save_dir = os.path.join('../results', opt.dataset, opt.model, opt.optimizer)
opt.save_dir = save_dir
if not os.path.exists(save_dir):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

res = dict()
res['opt'] = vars(opt)

train_dataset, test_dataset = get_datasets(opt.dataset)
# print('Dataset: ', train_dataset[0])
# exit()
train_loader, test_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True), DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)

model = get_model(opt.model, opt, res)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
if opt.optimizer == 'adam':
    optimizer = get_optimizer(opt.optimizer, opt)(model.parameters(), opt.lr)
elif opt.optimizer == 'sgd':
    optimizer = get_optimizer(opt.optimizer, opt)(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
elif opt.optimizer == 'adamw':
    optimizer = get_optimizer(opt.optimizer, opt)(model.parameters(), opt.lr)
elif opt.optimizer == 'spikefac_stab':
    optimizer = get_optimizer(opt.optimizer, opt)(model, opt.lr)
elif opt.optimizer == 'spikefac_stab_old':
    optimizer = get_optimizer(opt.optimizer, opt)(model, opt.lr)

loss_fn = CrossEntropyLoss()
trainer = Trainer(model, optimizer, loss_fn, train_loader, test_loader, opt, res)

trainer.train()
