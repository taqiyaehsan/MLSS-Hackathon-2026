from copy import deepcopy
import torch
from torch import nn, optim
from methods.BaseMethod import BaseMethod

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

class MyMethod(BaseMethod):
    def __init__(self, name):
        super().__init__(name)

    def run(self, net, retain_loader, forget_loader, val_loader):
        net.to(DEVICE)

        epochs = 1
        criterion_ce = nn.CrossEntropyLoss()
        criterion_kl = nn.KLDivLoss(reduction='batchmean')
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Precompute uniform distribution for forget batch size (dynamic inside loop)

        net.train()
        for ep in range(epochs):
            forget_iter = iter(forget_loader)
            retain_iter = iter(retain_loader)

            while True:
                try:
                    # Gradient ascent step on forget set with uniform KL loss
                    batch = next(forget_iter)
                    if isinstance(batch, dict):
                        inputs = batch["image"] if "image" in batch else batch.get("inputs", None)
                        targets = batch["age_group"] if "age_group" in batch else batch.get("labels", None)
                    else:
                        inputs, _ = batch
                    inputs = inputs.to(DEVICE)

                    optimizer.zero_grad()
                    outputs = net(inputs)
                    log_probs = nn.functional.log_softmax(outputs, dim=1)
                    uniform_dist = torch.full_like(outputs, 1.0 / outputs.size(1))
                    # Gradient ascent: maximize KL divergence to uniform = minimize negative KL
                    loss_kl = criterion_kl(log_probs, uniform_dist)
                    (-loss_kl).backward()  # ascent step
                    optimizer.step()

                    # Gradient descent step on retain set with CE
                    batch = next(retain_iter)
                    if isinstance(batch, dict):
                        inputs, targets = batch.get("image", None), batch.get("age_group", None)
                    else:
                        inputs, targets = batch
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss_ce = criterion_ce(outputs, targets)
                    loss_ce.backward()
                    optimizer.step()

                except StopIteration:
                    break
            scheduler.step()

        net.eval()
