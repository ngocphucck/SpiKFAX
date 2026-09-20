import matplotlib.pyplot as plt
import json
import os

from options import Options

opt = Options()
opt.dataset = ['MNIST']
opt.model = ['feedforward']
opt.optimizer = ['adam', 'adamw', 'sgd']

# For each combination of dataset and model, plot the train_loss, test_loss, train_accuracy, test_accuracy (all optimizers in 4 figure)
fig, axs = plt.subplots(len(opt.dataset) * len(opt.model), 4, figsize=(12, 10))
for i, dataset in enumerate(opt.dataset):
    for j, model in enumerate(opt.model):
        for k, optimizer in enumerate(opt.optimizer):
            save_dir = os.path.join('../results', f'{dataset}_{model}_{optimizer}_ce')
            with open(os.path.join(save_dir, 'results.json'), 'r') as f:
                res = json.load(f)
            axs[i * len(opt.model) + j, 0].plot(res['train_loss'], label=optimizer)
            axs[i * len(opt.model) + j, 0].set_title(f'{dataset} {model} Train Loss')
            axs[i * len(opt.model) + j, 1].plot(res['test_loss'], label=optimizer)
            axs[i * len(opt.model) + j, 1].set_title(f'{dataset} {model} Test Loss')
            axs[i * len(opt.model) + j, 2].plot(res['train_accuracy'], label=optimizer)
            axs[i * len(opt.model) + j, 2].set_title(f'{dataset} {model} Train Accuracy')
            axs[i * len(opt.model) + j, 3].plot(res['test_accuracy'], label=optimizer)
            axs[i * len(opt.model) + j, 3].set_title(f'{dataset} {model} Test Accuracy')

plt.tight_layout()
plt.savefig(os.path.join(f'../results', f'train_test_loss_accuracy_datasets_{opt.dataset}_{opt.model}_{opt.optimizer}.png'))
