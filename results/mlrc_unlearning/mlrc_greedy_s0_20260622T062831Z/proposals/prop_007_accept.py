from copy import deepcopy
import torch
from torch import nn, optim
from methods.BaseMethod import BaseMethod

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

def symmetric_kl(p_logits, q_logits):
    p = torch.softmax(p_logits, dim=1)
    q = torch.softmax(q_logits, dim=1)
    kl1 = torch.nn.functional.kl_div(torch.log(p), q, reduction='batchmean')
    kl2 = torch.nn.functional.kl_div(torch.log(q), p, reduction='batchmean')
    return (kl1 + kl2) / 2

class MyMethod(BaseMethod):
    def __init__(self, name):
        super().__init__(name)

    def run(self, net, retain_loader, forget_loader, val_loader):
        net.to(DEVICE)
        net.train()

        # Save original model predictions function for distillation
        with torch.no_grad():
            original_net = deepcopy(net).to(DEVICE)
            original_net.eval()

        # 1. Add small Gaussian noise to convolutional weights to disrupt memorization
        # Noise std is small to avoid catastrophic damage
        with torch.no_grad():
            for name, param in net.named_parameters():
                if 'conv' in name and param.requires_grad:
                    noise = torch.randn_like(param) * 0.05
                    param.add_(noise)

        # 2. Finetune on retain set with combined cross-entropy + symmetric KL divergence to original model
        criterion_ce = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)

        epochs = 3
        for epoch in range(epochs):
            net.train()
            for batch in retain_loader:
                if isinstance(batch, dict):
                    inputs = batch["image"] if "image" in batch else batch["inputs"]
                    targets = batch["age_group"] if "age_group" in batch else batch["targets"]
                else:
                    inputs, targets = batch
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                optimizer.zero_grad()
                outputs = net(inputs)

                with torch.no_grad():
                    orig_outputs = original_net(inputs)

                loss_ce = criterion_ce(outputs, targets)
                loss_kl = symmetric_kl(outputs, orig_outputs)

                loss = loss_ce + 0.5 * loss_kl
                loss.backward()
                optimizer.step()

        net.eval()
