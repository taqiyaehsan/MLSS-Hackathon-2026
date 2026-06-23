from copy import deepcopy
import torch
from torch import nn, optim
from methods.BaseMethod import BaseMethod

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

class MyMethod(BaseMethod):
    def __init__(self, name):
        super().__init__(name)

    def run(self, net, retain_loader, forget_loader, val_loader):
        """Unlearning implementation - add gradient ascent on forget set with uniform KL loss plus symmetric KL distillation on retain set

        Args:
            net: The model to be unlearned
            retain_loader: DataLoader for retained training data
            forget_loader: DataLoader for data to be forgotten
            val_loader: DataLoader for validation data

        Returns:
            The unlearned model
        """
        # Parameters
        ascent_epochs = 1  # few epochs ascent on forget set
        finetune_epochs = 1  # epochs finetuning on retain set
        lr_ascent = 0.001
        lr_finetune = 0.001
        kl_weight = 0.1

        criterion_ce = nn.CrossEntropyLoss()

        net_orig = deepcopy(net).to(DEVICE)
        net_orig.eval()

        # --- Phase 1: Gradient ascent on forget set to push predictions towards uniform ---
        optimizer_ascent = optim.SGD(net.parameters(), lr=lr_ascent, momentum=0.9, weight_decay=5e-4)

        def uniform_kl_loss(logits):
            # KL divergence between model predictions and uniform distribution
            log_prob = nn.functional.log_softmax(logits, dim=1)
            num_classes = logits.size(1)
            uniform_prob = torch.full_like(log_prob, 1.0 / num_classes)
            # kl_div input: log_prob, target probs
            return nn.functional.kl_div(log_prob, uniform_prob, reduction='batchmean')

        net.train()
        for _ in range(ascent_epochs):
            for sample in forget_loader:
                if isinstance(sample, dict):
                    inputs = sample["image"] if "image" in sample else sample[0]
                else:
                    inputs = sample[0]
                inputs = inputs.to(DEVICE)

                optimizer_ascent.zero_grad()
                outputs = net(inputs)
                loss = uniform_kl_loss(outputs)
                # Gradient ascent: maximize loss => minimize -loss
                (-loss).backward()
                optimizer_ascent.step()

        # --- Phase 2: Finetune on retain set with symmetric KL distillation ---
        optimizer = optim.SGD(net.parameters(), lr=lr_finetune, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=finetune_epochs)

        def symmetric_kl(p_logits, q_logits):
            p_log_prob = nn.functional.log_softmax(p_logits, dim=1)
            q_log_prob = nn.functional.log_softmax(q_logits, dim=1)
            p_prob = nn.functional.softmax(p_logits, dim=1)
            q_prob = nn.functional.softmax(q_logits, dim=1)
            kl_pq = nn.functional.kl_div(p_log_prob, q_prob, reduction='batchmean')
            kl_qp = nn.functional.kl_div(q_log_prob, p_prob, reduction='batchmean')
            return 0.5 * (kl_pq + kl_qp)

        for ep in range(finetune_epochs):
            net.train()
            for sample in retain_loader:
                if isinstance(sample, dict):
                    inputs = sample["image"] if "image" in sample else sample[0]
                    targets = sample["age_group"] if "age_group" in sample else sample[1]
                else:
                    inputs, targets = sample
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                optimizer.zero_grad()
                outputs = net(inputs)
                with torch.no_grad():
                    orig_outputs = net_orig(inputs)

                loss_ce = criterion_ce(outputs, targets)
                loss_kl = symmetric_kl(outputs, orig_outputs)
                loss = loss_ce + kl_weight * loss_kl

                loss.backward()
                optimizer.step()
            scheduler.step()

        net.eval()
