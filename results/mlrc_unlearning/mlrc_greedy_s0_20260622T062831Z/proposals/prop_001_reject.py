from copy import deepcopy
import torch
from torch import nn, optim
import torch.nn.functional as F
from methods.BaseMethod import BaseMethod

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

class MyMethod(BaseMethod):
    def __init__(self, name):
        super().__init__(name)

    def run(self, net, retain_loader, forget_loader, val_loader):
        # Save original model for distillation
        net.to(DEVICE)
        net_orig = deepcopy(net).to(DEVICE)
        net_orig.eval()
        
        epochs = 3  # small number for efficiency
        lr = 0.005

        criterion_ce = nn.CrossEntropyLoss()
        criterion_kl = nn.KLDivLoss(reduction='batchmean')

        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

        # Helper to get outputs with temperature scaling for forget set
        def uniform_target(logits):
            # uniform distribution over classes
            num_classes = logits.size(1)
            return torch.full_like(logits, 1.0 / num_classes)

        # Mix batches from retain_loader to speed up retain iteration
        retain_iter = iter(retain_loader)

        for ep in range(epochs):
            net.train()

            # Phase 1: Gradient ascent on forget set with uniform KL loss
            for batch in forget_loader:
                if isinstance(batch, dict):
                    inputs = batch['image'].to(DEVICE)
                else:
                    inputs, _ = batch
                    inputs = inputs.to(DEVICE)

                optimizer.zero_grad()
                outputs = net(inputs)
                log_probs = F.log_softmax(outputs, dim=1)
                uniform = uniform_target(outputs).to(DEVICE)
                loss_forget = -criterion_kl(log_probs, uniform)  # ascent = maximize KL divergence from uniform
                loss_forget.backward()
                optimizer.step()

            # Phase 2: Gradient descent on retain set with CE + distillation
            for batch in retain_loader:
                if isinstance(batch, dict):
                    inputs = batch['image'].to(DEVICE)
                    targets = batch['age_group'].to(DEVICE)
                else:
                    inputs, targets = batch
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                optimizer.zero_grad()
                outputs = net(inputs)
                loss_ce = criterion_ce(outputs, targets)

                with torch.no_grad():
                    outputs_orig = net_orig(inputs)
                    probs_orig = F.softmax(outputs_orig, dim=1)

                log_probs = F.log_softmax(outputs, dim=1)
                loss_distill = criterion_kl(log_probs, probs_orig) * 0.5  # weight distillation

                loss = loss_ce + loss_distill

                loss.backward()
                optimizer.step()

        net.eval()
