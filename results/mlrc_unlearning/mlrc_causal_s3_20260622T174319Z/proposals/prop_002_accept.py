import torch
from torch import nn, optim
import torch.nn.functional as F
from copy import deepcopy
from methods.BaseMethod import BaseMethod

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

class MyMethod(BaseMethod):
    def __init__(self, name):
        super().__init__(name)

    def run(self, net, retain_loader, forget_loader, val_loader):
        net.to(DEVICE)
        net.train()

        # Save original model for distillation
        net_orig = deepcopy(net).to(DEVICE)
        net_orig.eval()

        # Hyperparameters
        forget_steps = 3  # number of forget ascent epochs
        retain_epochs = 2  # number of retain finetune epochs
        lr_forget = 1e-3
        lr_retain = 1e-3
        temperature = 1.0

        # Uniform target distribution for forget set KL
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

        # Criterion for retain
        criterion_ce = nn.CrossEntropyLoss()

        # Helper symmetric KL divergence
        def sym_kl(p_logits, q_logits):
            p = F.log_softmax(p_logits / temperature, dim=1)
            q = F.log_softmax(q_logits / temperature, dim=1)
            p_soft = F.softmax(p_logits / temperature, dim=1)
            q_soft = F.softmax(q_logits / temperature, dim=1)
            kl_pq = F.kl_div(p, q_soft, reduction='batchmean')
            kl_qp = F.kl_div(q, p_soft, reduction='batchmean')
            return (kl_pq + kl_qp) / 2

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
                # KL from model outputs to uniform: KL(model || uniform) = sum p log(p/q)
                # Here q = uniform
                kl_loss = (F.softmax(outputs / temperature, dim=1) * (log_probs - torch.log(uniform_dist))).sum(dim=1).mean()
                # Gradient ascent -> maximize kl_loss, so minimize negative kl_loss
                loss = -kl_loss
                loss.backward()
                opt_forget.step()

        # --- Retain set fine-tuning phase with CE + symmetric KL distillation ---
        net.train()
        for epoch in range(retain_epochs):
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

                loss = loss_ce + 0.5 * loss_kl
                loss.backward()
                opt_retain.step()

        net.eval()
