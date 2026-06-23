from copy import deepcopy
import torch
from torch import nn, optim
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
        forget_ascent_epochs = 1  # single pass over forget set
        retain_finetune_epochs = 2  # few passes over retain set
        lr_forget = 0.01
        lr_retain = 0.001
        temperature = 0.1  # temperature for contrastive loss

        # Losses
        ce_loss = nn.CrossEntropyLoss()
        kl_loss = nn.KLDivLoss(reduction='batchmean')

        # Uniform target distribution for forget set
        n_classes = 10
        uniform_dist = torch.full((1, n_classes), 1.0 / n_classes, device=DEVICE)

        # Extract feature function (penultimate layer)
        def extract_features(x):
            # Assuming net has attribute 'fc' as final layer, extract penultimate features
            # For ResNet18, features before fc are from avgpool output flattened
            # We implement a forward hook to get features or redefine forward
            # Here we assume net has attribute 'avgpool' and 'fc'
            with torch.no_grad():
                # Replicate forward steps up to avgpool
                x = net.conv1(x)
                x = net.bn1(x)
                x = net.relu(x)
                x = net.maxpool(x)
                x = net.layer1(x)
                x = net.layer2(x)
                x = net.layer3(x)
                x = net.layer4(x)
                x = net.avgpool(x)
                feats = torch.flatten(x, 1)
            return feats

        # Prepare retain loader iterator for contrastive pairs
        retain_iter = iter(retain_loader)

        optimizer_forget = optim.SGD(net.parameters(), lr=lr_forget, momentum=0.9, weight_decay=5e-4)

        net.train()
        for _ in range(forget_ascent_epochs):
            for forget_batch in forget_loader:
                if isinstance(forget_batch, dict):
                    inputs_forget = forget_batch['image'].to(DEVICE)
                else:
                    inputs_forget, _ = forget_batch
                    inputs_forget = inputs_forget.to(DEVICE)

                # Get a batch from retain loader for contrastive loss
                try:
                    retain_batch = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_loader)
                    retain_batch = next(retain_iter)

                if isinstance(retain_batch, dict):
                    inputs_retain = retain_batch['image'].to(DEVICE)
                else:
                    inputs_retain, _ = retain_batch
                    inputs_retain = inputs_retain.to(DEVICE)

                optimizer_forget.zero_grad()

                outputs = net(inputs_forget)
                log_probs = nn.functional.log_softmax(outputs, dim=1)

                probs = torch.exp(log_probs)
                kl = torch.sum(probs * (torch.log(probs + 1e-10) - torch.log(uniform_dist + 1e-10)), dim=1).mean()

                # Contrastive feature loss: push forget features away from retain features
                feats_forget = extract_features(inputs_forget)  # no grad here is avoided since we optimize net
                feats_retain = extract_features(inputs_retain)

                # Normalize features
                feats_forget_norm = feats_forget / (feats_forget.norm(dim=1, keepdim=True) + 1e-10)
                feats_retain_norm = feats_retain / (feats_retain.norm(dim=1, keepdim=True) + 1e-10)

                # Compute similarity matrix: (forget_batch_size, retain_batch_size)
                sim_matrix = torch.matmul(feats_forget_norm, feats_retain_norm.t()) / temperature

                # For each forget sample, define contrastive loss to push away retain features
                # Use log-sum-exp over similarities as negative samples
                # We want to maximize distance, so minimize similarity
                # loss = mean logsumexp(similarities)

                contrastive_loss = torch.logsumexp(sim_matrix, dim=1).mean()

                # Total loss: negative KL (gradient ascent) + contrastive loss
                loss = -kl + contrastive_loss

                loss.backward()
                optimizer_forget.step()

        # --- Phase 2: Fine-tune on retain set with CE + symmetric KL distillation ---

        optimizer_retain = optim.SGD(net.parameters(), lr=lr_retain, momentum=0.9, weight_decay=5e-4)

        net.train()
        for epoch in range(retain_finetune_epochs):
            for batch in retain_loader:
                if isinstance(batch, dict):
                    inputs = batch['image'].to(DEVICE)
                    targets = batch['age_group'].to(DEVICE)
                else:
                    inputs, targets = batch
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                optimizer_retain.zero_grad()

                outputs = net(inputs)
                outputs_orig = net_orig(inputs)

                loss_ce = ce_loss(outputs, targets)

                log_probs = nn.functional.log_softmax(outputs, dim=1)
                log_probs_orig = nn.functional.log_softmax(outputs_orig, dim=1)

                # Symmetric KL divergence: KL(P||Q) + KL(Q||P)
                kl_pq = kl_loss(log_probs, log_probs_orig.exp().detach())
                kl_qp = kl_loss(log_probs_orig, log_probs.exp())
                loss_kl = kl_pq + kl_qp

                loss = loss_ce + loss_kl

                loss.backward()
                optimizer_retain.step()

        net.eval()
