from copy import deepcopy
import torch
from torch import nn, optim
from methods.BaseMethod import BaseMethod

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

def symmetric_kl(p_logits, q_logits):
    p = torch.softmax(p_logits, dim=1)
    q = torch.softmax(q_logits, dim=1)
    kl1 = torch.nn.functional.kl_div(torch.log(p + 1e-8), q, reduction='batchmean')
    kl2 = torch.nn.functional.kl_div(torch.log(q + 1e-8), p, reduction='batchmean')
    return (kl1 + kl2) / 2

class MyMethod(BaseMethod):
    def __init__(self, name):
        super().__init__(name)

    def run(self, net, retain_loader, forget_loader, val_loader):
        net.to(DEVICE)
        net.train()

        # Save original model for distillation
        with torch.no_grad():
            original_net = deepcopy(net).to(DEVICE)
            original_net.eval()

        # Selective reinitialization of conv filters with low L1 norm to increase plasticity
        with torch.no_grad():
            conv_weights = []
            conv_params = []
            for name, param in net.named_parameters():
                if 'conv' in name and param.requires_grad and param.dim() == 4:  # Conv weight shape [out_ch, in_ch, kH, kW]
                    conv_weights.append(param.view(param.size(0), -1))  # flatten per filter
                    conv_params.append(param)

            if conv_weights:
                all_filters = torch.cat(conv_weights, dim=0)  # (total_filters, filter_size)
                l1_norms = all_filters.abs().sum(dim=1)
                threshold = torch.quantile(l1_norms, 0.3)  # reinit bottom 30%
                idx_to_reinit = (l1_norms <= threshold).nonzero(as_tuple=False).squeeze(1)

                # Map indices back to parameters and reset
                start = 0
                for param in conv_params:
                    out_ch = param.size(0)
                    end = start + out_ch
                    reinit_idx = idx_to_reinit[(idx_to_reinit >= start) & (idx_to_reinit < end)] - start
                    if reinit_idx.numel() > 0:
                        # Reinit selected filters with Kaiming normal
                        for i in reinit_idx:
                            nn.init.kaiming_normal_(param[i])
                    start = end

        # Optimizers
        optimizer_forget = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        optimizer_retain = optim.SGD(net.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)

        criterion_ce = nn.CrossEntropyLoss()

        # Number of alternating epochs
        forget_epochs = 2
        retain_epochs = 2

        # Function to compute KL to uniform
        def kl_to_uniform(logits):
            prob = torch.softmax(logits, dim=1)
            uniform_prob = torch.full_like(prob, 1.0 / prob.size(1))
            return torch.nn.functional.kl_div(torch.log(prob + 1e-8), uniform_prob, reduction='batchmean')

        for cycle in range(3):  # Three alternating cycles
            # Phase 1: Gradient ascent on forget set to push predictions to uniform
            net.train()
            for epoch in range(forget_epochs):
                for batch in forget_loader:
                    if isinstance(batch, dict):
                        inputs = batch.get('image', batch.get('inputs'))
                        targets = batch.get('age_group', batch.get('targets', None))
                    else:
                        inputs, targets = batch
                    inputs = inputs.to(DEVICE)

                    optimizer_forget.zero_grad()
                    outputs = net(inputs)
                    loss = -kl_to_uniform(outputs)  # gradient ascent
                    loss.backward()
                    optimizer_forget.step()

            # Phase 2: Finetune on retain set with CE + symmetric KL distillation to original net
            net.train()
            for epoch in range(retain_epochs):
                for batch in retain_loader:
                    if isinstance(batch, dict):
                        inputs = batch.get('image', batch.get('inputs'))
                        targets = batch.get('age_group', batch.get('targets'))
                    else:
                        inputs, targets = batch
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                    optimizer_retain.zero_grad()
                    outputs = net(inputs)
                    with torch.no_grad():
                        orig_outputs = original_net(inputs)

                    loss_ce = criterion_ce(outputs, targets)
                    loss_kl = symmetric_kl(outputs, orig_outputs)
                    loss = loss_ce + 0.5 * loss_kl
                    loss.backward()
                    optimizer_retain.step()

        net.eval()
