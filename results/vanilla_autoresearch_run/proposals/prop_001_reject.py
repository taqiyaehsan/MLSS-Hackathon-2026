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
        Improved unlearning by performing gradient ascent steps on forget data
        simultaneously with gradient descent on retain data, to push forget data
        away while preserving retain knowledge.

        This is done in a single unified epoch with small learning rate and balanced
        steps per batch from each loader.
        """
        net.to(DEVICE)
        net.train()

        criterion = nn.CrossEntropyLoss()
        lr = 0.01
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

        # We will do 2 epochs of this combined ascent/descent to improve unlearning
        epochs = 2

        # To combine both loaders fairly, convert forget_loader to infinite iterator
        forget_iter = iter(forget_loader)

        for epoch in range(epochs):
            for retain_batch in retain_loader:
                # Get forget batch, reset iterator if exhausted
                try:
                    forget_batch = next(forget_iter)
                except StopIteration:
                    forget_iter = iter(forget_loader)
                    forget_batch = next(forget_iter)

                # Prepare retain data
                if isinstance(retain_batch, dict):
                    retain_inputs = retain_batch["image"].to(DEVICE)
                    retain_targets = retain_batch["age_group"].to(DEVICE)
                else:
                    retain_inputs, retain_targets = retain_batch
                    retain_inputs, retain_targets = retain_inputs.to(DEVICE), retain_targets.to(DEVICE)

                # Prepare forget data
                if isinstance(forget_batch, dict):
                    forget_inputs = forget_batch["image"].to(DEVICE)
                    forget_targets = forget_batch["age_group"].to(DEVICE)
                else:
                    forget_inputs, forget_targets = forget_batch
                    forget_inputs, forget_targets = forget_inputs.to(DEVICE), forget_targets.to(DEVICE)

                optimizer.zero_grad()

                # Forward + loss on retain (standard cross entropy)
                retain_outputs = net(retain_inputs)
                retain_loss = criterion(retain_outputs, retain_targets)

                # Forward + loss on forget (cross entropy)
                forget_outputs = net(forget_inputs)
                forget_loss = criterion(forget_outputs, forget_targets)

                # We want to minimize retain_loss but maximize forget_loss (to forget)
                # So total_loss = retain_loss - forget_loss
                total_loss = retain_loss - forget_loss

                total_loss.backward()
                optimizer.step()

        net.eval()
