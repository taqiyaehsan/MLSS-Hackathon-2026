from copy import deepcopy
import torch
from torch import nn, optim
from methods.BaseMethod import BaseMethod

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

class MyMethod(BaseMethod):
    def __init__(self, name):
        super().__init__(name)

    def run(self, net, retain_loader, forget_loader, val_loader):
        """Unlearning implementation - finetuning on retain set with symmetric KL distillation

        Args:
            net: The model to be unlearned
            retain_loader: DataLoader for retained training data
            forget_loader: DataLoader for data to be forgotten (unused here)
            val_loader: DataLoader for validation data

        Returns:
            The unlearned model
        """ 
        epochs = 1
        criterion_ce = nn.CrossEntropyLoss()
        # Prepare original model for distillation
        net_orig = deepcopy(net).to(DEVICE)
        net_orig.eval()

        optimizer = optim.SGD(net.parameters(), lr=0.001,
                              momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs)

        def symmetric_kl(p_logits, q_logits):
            # p_logits, q_logits are raw logits
            p_log_prob = nn.functional.log_softmax(p_logits, dim=1)
            q_log_prob = nn.functional.log_softmax(q_logits, dim=1)
            p_prob = nn.functional.softmax(p_logits, dim=1)
            q_prob = nn.functional.softmax(q_logits, dim=1)
            kl_pq = nn.functional.kl_div(p_log_prob, q_prob, reduction='batchmean')
            kl_qp = nn.functional.kl_div(q_log_prob, p_prob, reduction='batchmean')
            return 0.5 * (kl_pq + kl_qp)

        for ep in range(epochs):
            net.train()
            for batch_idx, sample in enumerate(retain_loader):
                if isinstance(sample, dict):
                    inputs = sample["image"]
                    targets = sample["age_group"]
                else:
                    inputs, targets = sample
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                optimizer.zero_grad()
                outputs = net(inputs)
                with torch.no_grad():
                    orig_outputs = net_orig(inputs)

                loss_ce = criterion_ce(outputs, targets)
                loss_kl = symmetric_kl(outputs, orig_outputs)

                # Weight for KL distillation small to keep retain accuracy stable
                loss = loss_ce + 0.1 * loss_kl

                loss.backward()
                optimizer.step()
            scheduler.step()

        net.eval()
