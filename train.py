import os
import numpy as np
import copy
import time
from collections import defaultdict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import argparser
from loss import UNetLoss
from utils import print_metrics

class Trainer(object):

    def __init__(self, config, model, dataset):
        self.config = config
        self.model = model
        self.dataset = dataset

        self.dataloader = {
            'train': DataLoader(self.dataset['train'], batch_size=self.config.batch_size, shuffle=True),
            'valid': DataLoader(self.dataset['valid'], batch_size=self.config.batch_size, shuffle=True),
        }

        os.makedirs(self.config.log_folder, exist_ok=True)
        os.makedirs(self.config.save_folder, exist_ok=True)

        self.device = torch.device(self.config.device if torch.cuda.is_available() else 'cpu')
        self.loss = UNetLoss()

        self.best_valid_loss = float("inf")
        self.best_model_wts = copy.deepcopy(self.model.state_dict())

        # optimizer
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.config.lr)
        self.exp_lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config.lr_scheduler_step, gamma=0.1)

        self.model.to(self.device)

    def train(self):
        num_epochs = self.config.epochs

        for epoch in range(num_epochs):
            # log
            print('Epoch {}/{}'.format(epoch, num_epochs-1))
            print('-' * 10)
            since = time.time()

            self.run_single_step(epoch)

            self.exp_lr_scheduler.step()

            # log
            time_elapsed = time.time() - since
            print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        print("The best valid loss is {:4f}".format(self.best_valid_loss))
        torch.save(self.model.state_dict(), self.config.save_folder + "\model-unet.pth")

    def run_single_step(self, epoch):
        loss_train, loss_valid = [], []

        for phase in ['train', 'valid']:
            if phase == 'train':
                self.model.train()
            else:
                self.model.eval()
            
            metrics = defaultdict(float)
            samples = 0
            
            for _, sample_batch in enumerate(self.dataloader[phase]):
                x, y_true = sample_batch
                x, y_true = x.float(), y_true.float()
                x, y_true = x.to(self.device), y_true.to(self.device)
                
                self.optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    y_pred = self.model(x)
                    _loss = self.loss.forward(y_pred, y_true, metrics=metrics)

                    if phase == 'valid':
                        loss_valid.append(_loss.item())
                        
                    # backward and optimize only in training phase
                    if phase == 'train':
                        loss_train.append(_loss.item())
                        _loss.backward()
                        self.optimizer.step()

                samples += x.size(0)

            print_metrics(metrics, samples, phase)
            epoch_loss = metrics['loss'] / samples
            
            # tensorboard            
            if phase == 'train':
                writer = SummaryWriter(log_dir=self.config.log_folder)
                writer.add_scalar('Loss/train', np.mean(loss_train), epoch)
            else:
                writer = SummaryWriter(log_dir=self.config.log_folder)
                writer.add_scalar('Loss/valid', np.mean(loss_valid), epoch)

                if epoch_loss < self.best_valid_loss:
                    print('saving best model with {:4f} better than {:4f}'.format(epoch_loss, self.best_valid_loss))
                    self.best_valid_loss = epoch_loss
                    self.best_model_wts = copy.deepcopy(self.model.state_dict())


def main():
    config, model, dataset = argparser()

    trainer = Trainer(config, model, dataset)

    trainer.train()


if __name__ == '__main__':
    main()



















