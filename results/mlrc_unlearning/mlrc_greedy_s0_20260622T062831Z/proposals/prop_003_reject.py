from copy import deepcopy
import torch
from torch import nn, optim
import torch.nn.functional as F
from methods.BaseMethod import BaseMethod

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

# Helper function for symmetric KL divergence

def symmetric_kl(p_logits, q_logits, T=1.0):
    p = F.log_softmax(p_logits / T, dim=1)
    q = F.log_softmax(q_logits / T, dim=1)
    p_soft = F.softmax(p_logits / T, dim=1)
    q_soft = F.softmax(q_logits / T, dim=1)
    kl_pq = F.kl_div(p, q_soft, reduction='batchmean')
    kl_qp = F.kl_div(q, p_soft, reduction='batchmean')
    return (kl_pq + kl_qp) / 2

# Contrastive loss between forget and retain features

def contrastive_loss(feat_forget, feat_retain, temperature=0.07):
    # Normalize features
    feat_forget = F.normalize(feat_forget, dim=1)
    feat_retain = F.normalize(feat_retain, dim=1)
    # Compute similarity matrix
    sim_matrix = torch.matmul(feat_forget, feat_retain.t()) / temperature
    # Maximize dissimilarity => maximize negative similarities
    # Use InfoNCE style loss with uniform negative targets (maximize -sim)
    labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
    loss_f2r = F.cross_entropy(sim_matrix, labels)
    return loss_f2r

class MyMethod(BaseMethod):
    def __init__(self, name):
        super().__init__(name)

    def run(self, net, retain_loader, forget_loader, val_loader):
        net.to(DEVICE)
        net.train()

        # Save a frozen copy of original model for distillation
        net_orig = deepcopy(net).eval().to(DEVICE)
        for p in net_orig.parameters():
            p.requires_grad = False

        # Optimizers
        forget_optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)
        retain_optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

        # Losses
        ce_loss = nn.CrossEntropyLoss()

        # Helper to extract penultimate features (before final FC)
        # Assume net has attribute 'fc' as last layer and 'avgpool' before it (resnet18)
        def extract_features(x):
            # Extract features before final FC
            # Adapted for resnet18
            with torch.no_grad():
                # We create a forward hook for features
                features = []

                def hook(module, input, output):
                    features.append(output.flatten(1))

                handle = net.avgpool.register_forward_hook(hook)
                _ = net(x)
                handle.remove()

            return features[0]

        # Phase 1: Gradient ascent on forget set to push predictions to uniform + contrastive loss
        epochs_forget = 2
        for epoch in range(epochs_forget):
            net.train()
            forget_iter = iter(forget_loader)
            retain_iter = iter(retain_loader)
            for _ in range(min(len(forget_loader), len(retain_loader))):
                try:
                    f_batch = next(forget_iter)
                except StopIteration:
                    break
                try:
                    r_batch = next(retain_iter)
                except StopIteration:
                    break

                # Unpack batches
                if isinstance(f_batch, dict):
                    f_inputs = f_batch["image"].to(DEVICE)
                else:
                    f_inputs, _ = f_batch
                    f_inputs = f_inputs.to(DEVICE)

                if isinstance(r_batch, dict):
                    r_inputs = r_batch["image"].to(DEVICE)
                else:
                    r_inputs, _ = r_batch
                    r_inputs = r_inputs.to(DEVICE)

                forget_optimizer.zero_grad()

                # Forward pass
                f_outputs = net(f_inputs)
                r_outputs = net(r_inputs)

                # KL to uniform for forget set (maximize entropy) via gradient ascent
                uniform_dist = torch.full_like(F.softmax(f_outputs, dim=1), 1.0 / f_outputs.size(1))
                log_prob = F.log_softmax(f_outputs, dim=1)
                kl_to_uniform = (uniform_dist * (uniform_dist.log() - log_prob)).sum(dim=1).mean()

                # Feature contrastive loss between forget and retain
                f_feats = extract_features(f_inputs)
                r_feats = extract_features(r_inputs)
                c_loss = contrastive_loss(f_feats, r_feats)

                # Total loss (maximize kl_to_uniform + c_loss) == gradient ascent, so minimize negative
                loss = -(kl_to_uniform + c_loss)
                loss.backward()
                forget_optimizer.step()

        # Phase 2: Fine-tune on retain set with CE + symmetric KL distillation to original model
        epochs_retain = 3
        for epoch in range(epochs_retain):
            net.train()
            for batch in retain_loader:
                if isinstance(batch, dict):
                    inputs = batch["image"].to(DEVICE)
                    targets = batch["age_group"].to(DEVICE)
                else:
                    inputs, targets = batch
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                retain_optimizer.zero_grad()

                outputs = net(inputs)
                with torch.no_grad():
                    orig_outputs = net_orig(inputs)

                loss_ce = ce_loss(outputs, targets)
                loss_kl = symmetric_kl(outputs, orig_outputs, T=1.0)

                loss = loss_ce + 0.5 * loss_kl
                loss.backward()
                retain_optimizer.step()

        net.eval()
