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
        net.to(DEVICE)
        net.train()

        # Save original model for distillation
        orig_net = deepcopy(net).to(DEVICE)
        orig_net.eval()

        # Reinitialize the last fully connected layer (fc) for partial forgetting
        if hasattr(net, 'fc') and hasattr(net.fc, 'reset_parameters'):
            net.fc.reset_parameters()
        elif hasattr(net, 'classifier') and hasattr(net.classifier, 'reset_parameters'):
            net.classifier.reset_parameters()

        # Optimizer for full net finetuning
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

        # Losses
        ce_loss = nn.CrossEntropyLoss()

        # Hyperparameters
        forget_epochs = 3
        retain_epochs = 1
        temperature = 2.0

        def kl_uniform_loss(outputs):
            # Encourage uniform predictions on forget set via KL divergence
            log_prob = F.log_softmax(outputs, dim=1)
            uniform_prob = torch.full_like(outputs, 1.0 / outputs.size(1))
            loss = F.kl_div(log_prob, uniform_prob, reduction='batchmean')
            return loss

        def symmetric_kl(p_logits, q_logits, temperature=1.0):
            p = F.log_softmax(p_logits / temperature, dim=1)
            q = F.softmax(q_logits / temperature, dim=1)
            kl1 = F.kl_div(p, q, reduction='batchmean')
            kl2 = F.kl_div(F.log_softmax(q_logits / temperature, dim=1), F.softmax(p_logits / temperature, dim=1), reduction='batchmean')
            return (kl1 + kl2) / 2

        # Move all loaders to device wrapper
        def to_device(batch):
            if isinstance(batch, dict):
                inputs = batch['image'].to(DEVICE)
                targets = batch['age_group'].to(DEVICE)
            else:
                inputs, targets = batch
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            return inputs, targets

        # Phase 1: Gradient ascent on forget set (maximize KL divergence to uniform) to forget
        for ep in range(forget_epochs):
            net.train()
            for batch in forget_loader:
                inputs, _ = to_device(batch)

                optimizer.zero_grad()
                outputs = net(inputs)

                loss = -kl_uniform_loss(outputs)  # Gradient ascent
                loss.backward()
                optimizer.step()

            # Phase 2: Gradient descent on retain set with CE + distillation regularization
            net.train()
            for batch in retain_loader:
                inputs, targets = to_device(batch)

                optimizer.zero_grad()
                outputs = net(inputs)
                with torch.no_grad():
                    orig_outputs = orig_net(inputs)

                loss_ce = ce_loss(outputs, targets)
                loss_kl = symmetric_kl(outputs, orig_outputs, temperature)

                loss = loss_ce + 0.5 * loss_kl
                loss.backward()
                optimizer.step()

        net.eval()
