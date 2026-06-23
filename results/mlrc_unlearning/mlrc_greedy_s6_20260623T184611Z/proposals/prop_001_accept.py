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

        # Parameters
        forget_ascent_epochs = 2
        retain_finetune_epochs = 2
        forget_lr = 0.001
        retain_lr = 0.001

        # KL divergence loss to uniform distribution for forget set ascent
        def uniform_target_kl(logits):
            # Target uniform distribution over classes
            num_classes = logits.shape[1]
            log_probs = F.log_softmax(logits, dim=1)
            uniform_dist = torch.full_like(log_probs, 1.0 / num_classes)
            # KL divergence from model output to uniform target
            # KL(P||Q) = sum P log(P/Q), here P=uniform, Q=model output
            # We want to maximize KL divergence from forget set data to original predictions,
            # but here we do gradient ascent to maximize loss pushing outputs towards uniform.
            # So compute negative KL: -KL(uniform||model) = sum uniform * (log uniform - log model)
            # But we implement loss = KL(model||uniform) since gradient ascent on it pushes logits towards uniform.
            # So use KL divergence from model output to uniform.
            kl = F.kl_div(log_probs, uniform_dist, reduction='batchmean')
            return kl

        # Symmetric KL divergence for distillation on retain set
        def symmetric_kl(p_logits, q_logits):
            p_log_prob = F.log_softmax(p_logits, dim=1)
            q_log_prob = F.log_softmax(q_logits, dim=1)
            p_prob = p_log_prob.exp()
            q_prob = q_log_prob.exp()
            kl_pq = F.kl_div(p_log_prob, q_prob, reduction='batchmean')
            kl_qp = F.kl_div(q_log_prob, p_prob, reduction='batchmean')
            return (kl_pq + kl_qp) / 2

        # Optimizers
        optimizer_forget = optim.SGD(net.parameters(), lr=forget_lr, momentum=0.9, weight_decay=5e-4)
        optimizer_retain = optim.SGD(net.parameters(), lr=retain_lr, momentum=0.9, weight_decay=5e-4)

        scheduler_retain = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_retain, T_max=retain_finetune_epochs)

        criterion = nn.CrossEntropyLoss()

        # Move data to device helper
        def prepare_batch(batch):
            if isinstance(batch, dict):
                inputs = batch.get("image", None)
                targets = batch.get("age_group", None)
            else:
                inputs, targets = batch
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            return inputs, targets

        # Step 1: Gradient ascent on forget set to push outputs toward uniform
        net.train()
        for epoch in range(forget_ascent_epochs):
            for batch in forget_loader:
                inputs, _ = prepare_batch(batch)
                optimizer_forget.zero_grad()
                outputs = net(inputs)
                loss_forget = uniform_target_kl(outputs)
                # Gradient ascent = maximize loss_forget
                (-loss_forget).backward()
                optimizer_forget.step()

        # Step 2: Fine-tune on retain set with CE + symmetric KL distillation to original model
        for epoch in range(retain_finetune_epochs):
            net.train()
            for batch in retain_loader:
                inputs, targets = prepare_batch(batch)
                optimizer_retain.zero_grad()
                outputs = net(inputs)
                with torch.no_grad():
                    orig_outputs = orig_net(inputs)

                loss_ce = criterion(outputs, targets)
                loss_kl = symmetric_kl(outputs, orig_outputs)
                loss = loss_ce + 5.0 * loss_kl  # distillation weight
                loss.backward()
                optimizer_retain.step()
            scheduler_retain.step()

        net.eval()
