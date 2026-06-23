from copy import deepcopy
import torch
from torch import nn, optim
from methods.BaseMethod import BaseMethod

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

class MyMethod(BaseMethod):
    def __init__(self, name):
        super().__init__(name)

    def run(self, net, retain_loader, forget_loader, val_loader):
        """Unlearning by alternating gradient ascent on forget data and gradient descent on retain data

        Args:
            net: The model to be unlearned
            retain_loader: DataLoader for retained training data
            forget_loader: DataLoader for data to be forgotten
            val_loader: DataLoader for validation data

        Modifies net in place and sets to eval mode at the end.
        """
        net.to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        lr = 0.001
        epochs = 2  # small number of epochs to keep quick

        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # To cycle through forget_loader independently
        forget_iter = iter(forget_loader)

        for epoch in range(epochs):
            net.train()

            # Alternate batches: one batch forget (gradient ascent), one batch retain (gradient descent)
            # until retain_loader exhausted
            for retain_batch in retain_loader:
                # Gradient ascent on one forget batch
                try:
                    forget_batch = next(forget_iter)
                except StopIteration:
                    forget_iter = iter(forget_loader)
                    forget_batch = next(forget_iter)

                # Process forget batch - gradient ascent on forget loss
                if isinstance(forget_batch, dict):
                    forget_inputs = forget_batch.get("image", None)
                    forget_targets = forget_batch.get("age_group", None)
                else:
                    forget_inputs, forget_targets = forget_batch
                forget_inputs = forget_inputs.to(DEVICE)
                forget_targets = forget_targets.to(DEVICE)

                optimizer.zero_grad()
                outputs_forget = net(forget_inputs)
                loss_forget = criterion(outputs_forget, forget_targets)
                # Gradient ascent to increase loss on forget data
                (-loss_forget).backward()
                optimizer.step()

                # Process retain batch - gradient descent on retain loss
                if isinstance(retain_batch, dict):
                    retain_inputs = retain_batch.get("image", None)
                    retain_targets = retain_batch.get("age_group", None)
                else:
                    retain_inputs, retain_targets = retain_batch
                retain_inputs = retain_inputs.to(DEVICE)
                retain_targets = retain_targets.to(DEVICE)

                optimizer.zero_grad()
                outputs_retain = net(retain_inputs)
                loss_retain = criterion(outputs_retain, retain_targets)
                loss_retain.backward()
                optimizer.step()

            scheduler.step()

        net.eval()
