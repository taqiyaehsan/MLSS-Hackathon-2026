from copy import deepcopy
import torch
from torch import nn, optim
from methods.BaseMethod import BaseMethod

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

class MyMethod(BaseMethod):
    def __init__(self, name):
        super().__init__(name)

    def run(self, net, retain_loader, forget_loader, val_loader):
        """Unlearning implementation - initial gradient ascent on forget set plus finetuning on retain set with symmetric KL distillation

        Args:
            net: The model to be unlearned
            retain_loader: DataLoader for retained training data
            forget_loader: DataLoader for data to be forgotten
            val_loader: DataLoader for validation data

        Returns:
            The unlearned model
        """
        # Parameters
        forget_ascent_epochs = 1
        retain_finetune_epochs = 1
        criterion_ce = nn.CrossEntropyLoss()

        net.to(DEVICE)
        net_orig = deepcopy(net).to(DEVICE)
        net_orig.eval()

        # Optimizer and scheduler for forgetting ascent
        optimizer_forget = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

        # Optimizer and scheduler for retain finetuning
        optimizer_retain = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        scheduler_retain = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_retain, T_max=retain_finetune_epochs)

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
            log_prob = nn.functional.log_softmax(logits, dim=1)
            num_classes = logits.shape[1]
            uniform_prob = torch.full_like(log_prob, 1.0 / num_classes)
            loss = nn.functional.kl_div(log_prob, uniform_prob, reduction='batchmean')
            return loss

        # Phase 1: Gradient ascent on forget set towards uniform predictions (to degrade forget info)
        net.train()
        for ep in range(forget_ascent_epochs):
            for batch_idx, sample in enumerate(forget_loader):
                if isinstance(sample, dict):
                    inputs = sample["image"] if "image" in sample else list(sample.values())[0]
                    targets = sample.get("age_group", None)  # not used
                else:
                    inputs, _ = sample
                inputs = inputs.to(DEVICE)

                optimizer_forget.zero_grad()
                outputs = net(inputs)
                loss = -uniform_kl_loss(outputs)  # gradient ascent: maximize KL to uniform means minimize negative
                loss.backward()
                optimizer_forget.step()

        # Phase 2: Finetune on retain set with cross-entropy + symmetric KL distillation to original model
        for ep in range(retain_finetune_epochs):
            net.train()
            for batch_idx, sample in enumerate(retain_loader):
                if isinstance(sample, dict):
                    inputs = sample["image"] if "image" in sample else list(sample.values())[0]
                    targets = sample.get("age_group", None)
                else:
                    inputs, targets = sample
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                optimizer_retain.zero_grad()
                outputs = net(inputs)
                with torch.no_grad():
                    orig_outputs = net_orig(inputs)

                loss_ce = criterion_ce(outputs, targets)
                loss_kl = symmetric_kl(outputs, orig_outputs)

                loss = loss_ce + 0.1 * loss_kl

                loss.backward()
                optimizer_retain.step()
            scheduler_retain.step()

        net.eval()
