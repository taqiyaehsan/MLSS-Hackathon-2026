from copy import deepcopy
import torch
from torch import nn, optim
from methods.BaseMethod import BaseMethod

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

class MyMethod(BaseMethod):
    def __init__(self, name):
        super().__init__(name)

    def run(self, net, retain_loader, forget_loader, val_loader):
        epochs = 2  # Increase epochs slightly for distillation regularization
        criterion = nn.CrossEntropyLoss()
        kl_loss_fn = nn.KLDivLoss(reduction='batchmean')

        # Keep original model frozen for distillation
        net.to(DEVICE)
        original_net = deepcopy(net).eval().to(DEVICE)
        for param in original_net.parameters():
            param.requires_grad = False

        optimizer = optim.SGD(net.parameters(), lr=0.001,
                              momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs)

        for ep in range(epochs):
            net.train()
            for batch_idx, sample in enumerate(retain_loader):
                if isinstance(sample, dict):
                    inputs = sample["image"] if "image" in sample else None
                    targets = sample["age_group"] if "age_group" in sample else None
                else:
                    inputs, targets = sample  # For CIFAR format
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                optimizer.zero_grad()

                outputs = net(inputs)
                loss_ce = criterion(outputs, targets)

                with torch.no_grad():
                    orig_outputs = original_net(inputs)

                log_probs = nn.functional.log_softmax(outputs, dim=1)
                orig_probs = nn.functional.softmax(orig_outputs, dim=1)

                # Symmetric KL divergence: KL(net||orig) + KL(orig||net)
                kl1 = kl_loss_fn(log_probs, orig_probs)
                kl2 = kl_loss_fn(nn.functional.log_softmax(orig_outputs, dim=1),
                                 nn.functional.softmax(outputs, dim=1))
                loss_kl = (kl1 + kl2) * 0.5

                loss = loss_ce + 0.5 * loss_kl  # Weighted sum

                loss.backward()
                optimizer.step()
            scheduler.step()

        net.eval()
