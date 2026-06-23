from copy import deepcopy
import torch
from torch import nn, optim
from methods.BaseMethod import BaseMethod

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

class MyMethod(BaseMethod):
    def __init__(self, name):
        super().__init__(name)

    def run(self, net, retain_loader, forget_loader, val_loader):
        """Unlearning implementation combining lightweight gradient ascent on forget set with symmetric KL distillation on retain set

        Args:
            net: The model to be unlearned
            retain_loader: DataLoader for retained training data
            forget_loader: DataLoader for data to be forgotten
            val_loader: DataLoader for validation data

        Returns:
            The unlearned model
        """
        epochs_retrain = 1
        ascent_epochs = 1
        criterion_ce = nn.CrossEntropyLoss()

        net_orig = deepcopy(net).to(DEVICE)
        net_orig.eval()

        optimizer_retrain = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        scheduler_retrain = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_retrain, T_max=epochs_retrain)

        def symmetric_kl(p_logits, q_logits):
            p_log_prob = nn.functional.log_softmax(p_logits, dim=1)
            q_log_prob = nn.functional.log_softmax(q_logits, dim=1)
            p_prob = nn.functional.softmax(p_logits, dim=1)
            q_prob = nn.functional.softmax(q_logits, dim=1)
            kl_pq = nn.functional.kl_div(p_log_prob, q_prob, reduction='batchmean')
            kl_qp = nn.functional.kl_div(q_log_prob, p_prob, reduction='batchmean')
            return 0.5 * (kl_pq + kl_qp)

        def uniform_kl_loss(logits):
            # KL divergence from model output to uniform distribution
            log_probs = nn.functional.log_softmax(logits, dim=1)
            n_classes = logits.size(1)
            uniform_prob = torch.full_like(log_probs, 1.0 / n_classes)
            # kl_div input is log_prob, target is prob
            loss = nn.functional.kl_div(log_probs, uniform_prob, reduction='batchmean')
            return loss

        # Phase 1: Gradient ascent on forget set to push predictions towards uniform
        net.train()
        optimizer_forget = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        # We do gradient ascent, so will minimize negative uniform_kl_loss
        for ep in range(ascent_epochs):
            for batch_idx, sample in enumerate(forget_loader):
                if isinstance(sample, dict):
                    inputs = sample["image"] if "image" in sample else sample["inputs"]
                    targets = sample["age_group"] if "age_group" in sample else None
                else:
                    inputs, _ = sample
                inputs = inputs.to(DEVICE)

                optimizer_forget.zero_grad()
                outputs = net(inputs)

                loss = uniform_kl_loss(outputs)

                # Gradient ascent step: maximize uniform KL loss
                # So minimize -loss
                (-loss).backward()
                optimizer_forget.step()

        # Phase 2: Fine-tune on retain set with cross-entropy + symmetric KL distillation
        for ep in range(epochs_retrain):
            net.train()
            for batch_idx, sample in enumerate(retain_loader):
                if isinstance(sample, dict):
                    inputs = sample["image"] if "image" in sample else sample["inputs"]
                    targets = sample["age_group"] if "age_group" in sample else None
                else:
                    inputs, targets = sample
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                optimizer_retrain.zero_grad()
                outputs = net(inputs)
                with torch.no_grad():
                    orig_outputs = net_orig(inputs)

                loss_ce = criterion_ce(outputs, targets)
                loss_kl = symmetric_kl(outputs, orig_outputs)

                loss = loss_ce + 0.1 * loss_kl

                loss.backward()
                optimizer_retrain.step()
            scheduler_retrain.step()

        net.eval()
