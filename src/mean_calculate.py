import numpy
import os
import json


res_dir = '../results'
# dataset = 'MNIST'
# dataset = 'F-MNIST'
dataset = 'N-MNIST'
# dataset = 'CIFAR100'
model_name = 'resnet18'
optimizer = 'adamw'
lr = 1e-5
n_epochs = 30

seeds = [0, 1]

a = []
for seed in seeds:
    fn = f'results_lr_{lr}_seed_{seed}.json'
    dp = os.path.join(res_dir, dataset, model_name, optimizer, fn)

    with open(dp, 'r') as f:
        res = json.load(f)

    test_acc = res['test_accuracy'][:n_epochs + 1]
    a.append(max(test_acc))

print('a: ', a)
print('Mean: ', numpy.mean(a))
print('Std: ', numpy.std(a))