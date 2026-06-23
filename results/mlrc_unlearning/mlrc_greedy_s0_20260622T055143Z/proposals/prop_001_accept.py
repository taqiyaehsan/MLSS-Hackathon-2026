from copy import deepcopy
import torch
from torch import nn, optim
import torch.nn.functional as F
from methods.BaseMethod import BaseMethod

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

class MyMethod(BaseMethod):
    def __init__(self, name):
        super().__init__(name)

    def _get_features(self, net, x):
        # Extract features before final fc layer for contrastive loss
        # Assume net is ResNet18 or similar
        # We use the penultimate layer output (avgpool output flattened)
        net.eval()
        with torch.no_grad():
            # forward until avgpool
            out = x.to(DEVICE)
            for name, module in net.named_children():
                if name == 'fc':
                    break
                out = module(out)
            features = out.view(out.size(0), -1)
        return features

    def run(self, net, retain_loader, forget_loader, val_loader):
        net.to(DEVICE)
        net.train()

        # Save original net for reference predictions
        net_orig = deepcopy(net).to(DEVICE)
        net_orig.eval()

        criterion_ce = nn.CrossEntropyLoss()
        criterion_kl = nn.KLDivLoss(reduction='batchmean')

        # Optimizer for unlearning
        optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)

        # 1) One epoch over forget set: gradient ascent on KL divergence to uniform distribution (increase entropy)
        # 2) 4 epochs alternating: gradient ascent on contrastive loss between forget and retain features + gradient descent on retain CE loss

        # Step 1: KL divergence to uniform on forget set
        net.train()
        uniform_dist = torch.full((retain_loader.dataset.classes if hasattr(retain_loader.dataset, 'classes') else 10,), 1.0/10, device=DEVICE)

        with torch.enable_grad():
            for inputs, targets in forget_loader:
                inputs = inputs.to(DEVICE)
                optimizer.zero_grad()
                outputs = net(inputs)
                log_probs = F.log_softmax(outputs, dim=1)
                # Target uniform distribution
                target_uniform = uniform_dist.unsqueeze(0).expand_as(log_probs)
                loss_kl = -criterion_kl(log_probs, target_uniform)  # ascend to max entropy
                loss_kl.backward()
                optimizer.step()

        # Step 2: multiple epochs alternating contrastive ascent on forget vs retain, and CE descent on retain
        epochs = 4
        temperature = 0.5

        retain_iter = iter(retain_loader)
        forget_iter = iter(forget_loader)

        for ep in range(epochs):
            net.train()
            # Contrastive ascent step on forget set
            for _ in range(len(forget_loader)):
                try:
                    f_inputs, _ = next(forget_iter)
                except StopIteration:
                    forget_iter = iter(forget_loader)
                    f_inputs, _ = next(forget_iter)
                try:
                    r_inputs, _ = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_loader)
                    r_inputs, _ = next(retain_iter)

                f_inputs = f_inputs.to(DEVICE)
                r_inputs = r_inputs.to(DEVICE)

                optimizer.zero_grad()

                # Get features
                f_feats = net._modules.get('avgpool')(net._modules.get('layer4')(net._modules.get('layer3')(net._modules.get('layer2')(net._modules.get('layer1')(net._modules.get('relu')(net._modules.get('bn1')(net._modules.get('conv1')(f_inputs))))))))
                f_feats = torch.flatten(f_feats, 1)
                r_feats = net._modules.get('avgpool')(net._modules.get('layer4')(net._modules.get('layer3')(net._modules.get('layer2')(net._modules.get('layer1')(net._modules.get('relu')(net._modules.get('bn1')(net._modules.get('conv1')(r_inputs))))))))
                r_feats = torch.flatten(r_feats, 1)

                # Normalize features
                f_feats = F.normalize(f_feats, dim=1)
                r_feats = F.normalize(r_feats, dim=1)

                # Compute similarity matrix
                sim_matrix = torch.mm(f_feats, r_feats.t()) / temperature

                # Contrastive loss: maximize distance (gradient ascent) between forget and retain features
                # So we minimize -sim_matrix mean
                loss_contrastive = -sim_matrix.mean()

                loss_contrastive.backward()
                optimizer.step()

            # After contrastive ascent, do one epoch CE training on retain set
            net.train()
            for r_inputs, r_targets in retain_loader:
                r_inputs, r_targets = r_inputs.to(DEVICE), r_targets.to(DEVICE)
                optimizer.zero_grad()
                outputs = net(r_inputs)
                loss_ce = criterion_ce(outputs, r_targets)
                loss_ce.backward()
                optimizer.step()

        net.eval()
