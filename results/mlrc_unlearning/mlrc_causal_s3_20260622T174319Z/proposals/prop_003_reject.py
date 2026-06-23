import torch
from torch import nn, optim
import torch.nn.functional as F
from copy import deepcopy
from methods.BaseMethod import BaseMethod

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

class MyMethod(BaseMethod):
    def __init__(self, name):
        super().__init__(name)

    def compute_fisher(self, net, data_loader, criterion_ce, num_samples=1000):
        net.eval()
        fisher = {}
        for n, p in net.named_parameters():
            fisher[n] = torch.zeros_like(p, device=DEVICE)

        n_samples = 0
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, dict):
                    inputs = batch['image'].to(DEVICE)
                    targets = batch['age_group'].to(DEVICE)
                else:
                    inputs, targets = batch
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                outputs = net(inputs)
                loss = criterion_ce(outputs, targets)
                net.zero_grad()
                loss.backward()

                # Accumulate squared gradients
                for n, p in net.named_parameters():
                    if p.grad is not None:
                        fisher[n] += (p.grad.detach() ** 2) * inputs.size(0)
                n_samples += inputs.size(0)
                if n_samples >= num_samples:
                    break

        # Normalize
        for n in fisher:
            fisher[n] /= n_samples
        net.train()
        return fisher

    def ewc_loss(self, net, net_orig, fisher, ewc_lambda):
        loss = 0.0
        for n, p in net.named_parameters():
            p0 = dict(net_orig.named_parameters())[n]
            loss += (fisher[n] * (p - p0).pow(2)).sum()
        return ewc_lambda * loss

    def run(self, net, retain_loader, forget_loader, val_loader):
        net.to(DEVICE)
        net.train()

        # Save original model for distillation
        net_orig = deepcopy(net).to(DEVICE)
        net_orig.eval()

        # Hyperparameters
        forget_steps = 3  # forget ascent epochs
        retain_epochs = 2  # retain finetune epochs
        lr_forget = 1e-3
        lr_retain = 5e-4
        temperature = 1.0
        ewc_lambda = 1000.0  # moderate penalty

        # Determine num_classes from forget_loader
        num_classes = None
        for batch in forget_loader:
            if isinstance(batch, dict):
                _, targets = batch.get('image'), batch.get('age_group')
            else:
                _, targets = batch
            num_classes = targets.max().item() + 1
            break
        uniform_dist = torch.full((num_classes,), 1.0 / num_classes, device=DEVICE)

        # Optimizers
        opt_forget = optim.SGD(net.parameters(), lr=lr_forget, momentum=0.9, weight_decay=5e-4)
        opt_retain = optim.SGD(net.parameters(), lr=lr_retain, momentum=0.9, weight_decay=5e-4)

        criterion_ce = nn.CrossEntropyLoss()

        def sym_kl(p_logits, q_logits):
            p_log = F.log_softmax(p_logits / temperature, dim=1)
            q_log = F.log_softmax(q_logits / temperature, dim=1)
            p_soft = F.softmax(p_logits / temperature, dim=1)
            q_soft = F.softmax(q_logits / temperature, dim=1)
            kl_pq = F.kl_div(p_log, q_soft, reduction='batchmean')
            kl_qp = F.kl_div(q_log, p_soft, reduction='batchmean')
            return (kl_pq + kl_qp) * 0.5

        # --- Compute Fisher Information on retain set (1 epoch, limited samples) ---
        fisher = self.compute_fisher(net, retain_loader, criterion_ce, num_samples=1000)

        # --- Forget set gradient ascent phase ---
        net.train()
        for _ in range(forget_steps):
            for batch in forget_loader:
                opt_forget.zero_grad()
                if isinstance(batch, dict):
                    inputs = batch['image'].to(DEVICE)
                else:
                    inputs = batch[0].to(DEVICE)

                outputs = net(inputs)
                log_probs = F.log_softmax(outputs / temperature, dim=1)
                # KL(model||uniform) = sum p log(p/q), q=uniform
                kl_loss = (F.softmax(outputs / temperature, dim=1) * (log_probs - torch.log(uniform_dist))).sum(dim=1).mean()
                loss = -kl_loss  # gradient ascent
                loss.backward()
                opt_forget.step()

        # --- Retain set fine-tuning phase with CE + symmetric KL + EWC penalty ---
        net.train()
        for _ in range(retain_epochs):
            for batch in retain_loader:
                opt_retain.zero_grad()
                if isinstance(batch, dict):
                    inputs = batch['image'].to(DEVICE)
                    targets = batch['age_group'].to(DEVICE)
                else:
                    inputs, targets = batch
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                outputs = net(inputs)
                with torch.no_grad():
                    outputs_orig = net_orig(inputs)

                loss_ce = criterion_ce(outputs, targets)
                loss_kl = sym_kl(outputs, outputs_orig)
                loss_ewc = self.ewc_loss(net, net_orig, fisher, ewc_lambda)

                loss = loss_ce + 0.5 * loss_kl + loss_ewc
                loss.backward()
                opt_retain.step()

        net.eval()
