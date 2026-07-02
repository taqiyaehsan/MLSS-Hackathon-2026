from copy import deepcopy
import torch
from torch import nn, optim
from methods.BaseMethod import BaseMethod

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

class MyMethod(BaseMethod):
    def __init__(self, name):
        super().__init__(name)

    def run(self, net, retain_loader, forget_loader, val_loader):
        """
        Unlearning by alternating gradient ascent on the forget set (to unlearn) 
        and gradient descent on the retain set (to keep accuracy), with a small learning rate.

        Args:
            net: The model to be unlearned
            retain_loader: DataLoader for retained training data
            forget_loader: DataLoader for data to be forgotten
            val_loader: DataLoader for validation data
        """
        net.to(DEVICE)
        criterion = nn.CrossEntropyLoss()

        # Hyperparameters
        epochs = 2
        lr = 1e-3
        forget_lambda = 1.0  # weight of forget ascent loss

        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

        # Convert forget_loader and retain_loader to iterators for alternating batches
        retain_iter = iter(retain_loader)
        forget_iter = iter(forget_loader)

        net.train()
        for epoch in range(epochs):
            # We alternate batches: one from forget_loader (for ascent), one from retain_loader (for descent)
            # To keep it balanced, run for max length of longer loader
            max_batches = max(len(retain_loader), len(forget_loader))

            for _ in range(max_batches):
                # Gradient ascent on forget set to unlearn
                try:
                    sample_forget = next(forget_iter)
                except StopIteration:
                    forget_iter = iter(forget_loader)
                    sample_forget = next(forget_iter)

                if isinstance(sample_forget, dict):
                    inputs_f = sample_forget["image"]
                    targets_f = sample_forget["age_group"]
                else:
                    inputs_f, targets_f = sample_forget
                inputs_f, targets_f = inputs_f.to(DEVICE), targets_f.to(DEVICE)

                optimizer.zero_grad()
                outputs_f = net(inputs_f)
                loss_f = criterion(outputs_f, targets_f)

                # Gradient ascent: maximize loss on forget set
                (-forget_lambda * loss_f).backward()
                optimizer.step()

                # Gradient descent on retain set to keep performance
                try:
                    sample_retain = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_loader)
                    sample_retain = next(retain_iter)

                if isinstance(sample_retain, dict):
                    inputs_r = sample_retain["image"]
                    targets_r = sample_retain["age_group"]
                else:
                    inputs_r, targets_r = sample_retain
                inputs_r, targets_r = inputs_r.to(DEVICE), targets_r.to(DEVICE)

                optimizer.zero_grad()
                outputs_r = net(inputs_r)
                loss_r = criterion(outputs_r, targets_r)

                loss_r.backward()
                optimizer.step()

        net.eval()
