import os
import json
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

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

opt.n_epochs = 20
opt.batch_size = 64
opt.lr = 1e-4
opt.alpha = 0.95

opt.num_inputs = 28 * 28
opt.num_hidden = 128
opt.num_outputs = 10
opt.beta = 0.9
opt.n_steps = 25

def parse_args():
    parser = argparse.ArgumentParser(description='Train a spiking neural network on MNIST.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset to use (MNIST or CIFAR10).')
    parser.add_argument('--seed', type=int, default=0, help='Seed')

    return parser.parse_args()

params = parse_args()

opt.lr = params.lr
opt.dataset = params.dataset
opt.seed = params.seed
if opt.dataset == 'MNIST':
    opt.num_inputs = 28 * 28
    opt.num_outputs = 10
elif opt.dataset == 'F-MNIST':
    opt.num_inputs = 28 * 28
    opt.num_outputs = 10
elif opt.dataset == 'CIFAR10':
    opt.num_inputs = 32 * 32 * 3
    opt.num_outputs = 10
elif opt.dataset == 'CIFAR100':
    opt.num_inputs = 32 * 32 * 3
    opt.num_outputs = 100

save_dir = os.path.join('../results', opt.dataset, opt.model, opt.optimizer)
opt.save_dir = save_dir
if not os.path.exists(save_dir):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

res = dict()
res['opt'] = vars(opt)

train_dataset, test_dataset = get_datasets(opt.dataset)
train_loader, test_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True), DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)

model = get_model(opt.model, opt, res)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = get_optimizer(opt.optimizer, opt)(model, opt.lr)
loss_fn = CrossEntropyLoss()
trainer = Trainer(model, optimizer, loss_fn, train_loader, test_loader, opt, res)

trainer.train()

with open(os.path.join(save_dir, f'results_lr_{opt.lr}_seed_{opt.seed}.json'), 'w') as f:
    json.dump(res, f)
