import json

from tqdm import tqdm 
import matplotlib.pyplot as plt
import os
import torch


def rate_encode(x, T):
    """
    x: [B, C, H, W], values in [0, 1]
    returns: [T, B, C, H, W]
    """
    return (
        torch.rand((T, *x.shape), device=x.device) < x.unsqueeze(0)
    ).float()


class Trainer:
    def __init__(self, model, optimizer, loss_fn, train_loader, test_loader, opt, res):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.opt = opt
        self.res = res

    def train(self):
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to('cuda' if torch.cuda.is_available() else 'cpu').float()
            target = target.to('cuda' if torch.cuda.is_available() else 'cpu')
            if self.opt.encode:
                data_spikes = rate_encode(data, self.opt.n_steps)
            else:
                data_spikes = data.transpose(0, 1)
            with torch.no_grad():
                spk2_rec = self.model(data_spikes)
            spike_counts = spk2_rec.sum(dim=1)
            loss = self.loss_fn(spike_counts, target)
            train_loss += loss.item()
        avg_train_loss = train_loss / len(self.train_loader)
        if 'train_loss' not in self.res:
            self.res['train_loss'] = []
        self.res['train_loss'].append(avg_train_loss)
        self.evaluate()

        for epoch in range(self.opt.n_epochs):
            train_loss = 0.0
            pbar = tqdm(self.train_loader, desc=f"Epoch [{epoch+1}/{self.opt.n_epochs}]")
            self.model.train()
            for batch_idx, (data, target) in enumerate(pbar):
                data = data.to('cuda' if torch.cuda.is_available() else 'cpu').float()
                target = target.to('cuda' if torch.cuda.is_available() else 'cpu')
                if self.opt.encode:
                    data_spikes = rate_encode(data, self.opt.n_steps)
                else:
                    data_spikes = data.transpose(0, 1)
                # print('Min data:', data.min().item(), 'Max data:', data.max().item())
                # print('Data shape:', data.shape)
                # print('Target shape:', target.shape)
                self.optimizer.zero_grad()
                spk2_rec = self.model(data_spikes)
                # print('spk2_rec shape:', spk2_rec.shape)
                # exit()
                spike_counts = spk2_rec.sum(dim=1)
                # print('spk2_rec shape:', spk2_rec.shape)
                # print('min spk2_rec:', spk2_rec.min().item())
                # print('max spk2_rec:', spk2_rec.max().item())
                # print('spk1_rec shape:', spk1_rec.shape)
                # print('min spk1_rec:', spk1_rec.min().item())
                # print('max spk1_rec:', spk1_rec.max().item())
                # print('Target shape:', target.shape)
                # print('min target:', target.min().item())
                # print('max target:', target.max().item())
                # print('spike_counts shape:', spike_counts.shape)
                # exit()
                loss = self.loss_fn(spike_counts, target)
                # for step in range(self.opt.n_steps):
                    # loss += self.loss_fn(spk2_rec[step], target)

                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                if 'step_loss' not in self.res:
                    self.res['step_loss'] = []
                self.res['step_loss'].append(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            avg_train_loss = train_loss / len(self.train_loader)
            if 'train_loss' not in self.res:
                self.res['train_loss'] = []
            self.res['train_loss'].append(avg_train_loss)
            self.evaluate()

            with open(os.path.join(self.opt.save_dir, f'results_lr_{self.opt.lr}_seed_{self.opt.seed}.json'), 'w') as f:
                json.dump(self.res, f)

    def evaluate(self):
        self.model.eval()
        test_loss = 0.0
        correct = 0
        top_5_correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.to('cuda' if torch.cuda.is_available() else 'cpu').float()
                target = target.to('cuda' if torch.cuda.is_available() else 'cpu')
                if self.opt.encode:
                    data_spikes = rate_encode(data, self.opt.n_steps)
                else:
                    data_spikes = data.transpose(0, 1)
                with torch.no_grad():
                    spk2_rec = self.model(data_spikes)
                spike_counts = spk2_rec.sum(dim=1)
                loss = self.loss_fn(spike_counts, target)
                test_loss += loss.item()
                pred = spike_counts.argmax(dim=1)
                top_5_pred = spike_counts.topk(5, dim=1).indices
                correct += pred.eq(target).sum().item()
                top_5_correct += sum([1 if target[i] in top_5_pred[i] else 0 for i in range(len(target))])

        avg_test_loss = test_loss / len(self.test_loader)
        accuracy = 100.0 * correct / len(self.test_loader.dataset)
        top_5_test_accuracy = 100.0 * top_5_correct / len(self.test_loader.dataset)
        if 'test_loss' not in self.res:
            self.res['test_loss'] = []
        self.res['test_loss'].append(avg_test_loss)
        if 'test_accuracy' not in self.res:
            self.res['test_accuracy'] = []
        self.res['test_accuracy'].append(accuracy)
        if 'top_5_test_accuracy' not in self.res:
            self.res['top_5_test_accuracy'] = []
        self.res['top_5_test_accuracy'].append(top_5_test_accuracy)

        correct = 0
        top_5_correct = 0
        with torch.no_grad():
            for data, target in self.train_loader:
                data = data.to('cuda' if torch.cuda.is_available() else 'cpu').float()
                target = target.to('cuda' if torch.cuda.is_available() else 'cpu')
                if self.opt.encode:
                    data_spikes = rate_encode(data, self.opt.n_steps)
                else:
                    data_spikes = data.transpose(0, 1)
                with torch.no_grad():
                    spk2_rec = self.model(data_spikes)
                spike_counts = spk2_rec.sum(dim=1)
                pred = spike_counts.argmax(dim=1)
                top_5_pred = spike_counts.topk(5, dim=1).indices
                correct += pred.eq(target).sum().item()
                top_5_correct += sum([1 if target[i] in top_5_pred[i] else 0 for i in range(len(target))])

        accuracy = 100.0 * correct / len(self.train_loader.dataset)
        top_5_train_accuracy = 100.0 * top_5_correct / len(self.train_loader.dataset)
        if 'train_accuracy' not in self.res:
            self.res['train_accuracy'] = []
        self.res['train_accuracy'].append(accuracy)
        if 'top_5_train_accuracy' not in self.res:
            self.res['top_5_train_accuracy'] = []
        self.res['top_5_train_accuracy'].append(top_5_train_accuracy)

        # Plot train_loss, test_loss, train_accuracy, test_accuracy
        fig, axs = plt.subplots(2, 4, figsize=(16, 8))
        axs[0, 0].plot(self.res['train_loss'], label=f'Train Loss', marker='o')
        axs[0, 0].set_title(f'Train Loss(min: {min(self.res["train_loss"]):.4f})')
        axs[1, 0].plot(self.res['test_loss'], label=f'Test Loss', marker='o')
        axs[1, 0].set_title(f'Test Loss(min: {min(self.res["test_loss"]):.4f})')
        axs[0, 1].plot(self.res['train_accuracy'], label=f'Train Accuracy', marker='o')
        axs[0, 1].set_title(f'Train Accuracy(max: {max(self.res["train_accuracy"]):.4f})')
        axs[1, 1].plot(self.res['test_accuracy'], label=f'Test Accuracy', marker='o')
        axs[1, 1].set_title(f'Test Accuracy(max: {max(self.res["test_accuracy"]):.4f})')
        axs[0, 2].plot(self.res['top_5_train_accuracy'], label=f'Top-5 Train Accuracy', marker='o')
        axs[0, 2].set_title(f'Top-5 Train Accuracy(max: {max(self.res["top_5_train_accuracy"]):.4f})')
        axs[1, 2].plot(self.res['top_5_test_accuracy'], label=f'Top-5 Test Accuracy', marker='o')
        axs[1, 2].set_title(f'Top-5 Test Accuracy(max: {max(self.res["top_5_test_accuracy"]):.4f})')
        if 'step_loss' in self.res and len(self.res['step_loss']) > 0:
            axs[0, 3].plot(self.res['step_loss'], label='Step Loss')
            axs[0, 3].set_title(f'Step Loss(max: {min(self.res["step_loss"]):.4f})')

        for ax in axs.flat:
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Value')
            ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.opt.save_dir, f'results_lr_{self.opt.lr}_seed_{self.opt.seed}.png'))
