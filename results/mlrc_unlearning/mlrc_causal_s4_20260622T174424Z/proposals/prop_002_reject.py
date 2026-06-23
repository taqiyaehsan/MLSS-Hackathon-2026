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

        # Save original model for reference if needed (not used here but kept for extensibility)
        original_net = deepcopy(net).to(DEVICE)
        original_net.eval()

        # Loss and optimizer for forget set ascent
        kl_loss_fn = nn.KLDivLoss(reduction='batchmean')

        # Build uniform target distribution over classes assuming CIFAR-10 classes (10 classes)
        num_classes = 10
        uniform_prob = torch.full((1, num_classes), 1.0 / num_classes, device=DEVICE)

        # Optimizer for forget set ascent
        forget_optimizer = optim.SGD(net.parameters(), lr=0.003, momentum=0.9, weight_decay=5e-4)

        # A few epochs of gradient ascent on forget set to push predictions toward uniform
        forget_epochs = 2
        for ep in range(forget_epochs):
            for batch in forget_loader:
                if isinstance(batch, dict):
                    inputs = batch["image"] if "image" in batch else batch.get("inputs", None)
                    targets = batch.get("age_group", None)
                    if inputs is None:
                        inputs = batch["inputs"] if "inputs" in batch else None
                else:
                    inputs, targets = batch
                inputs = inputs.to(DEVICE)

                forget_optimizer.zero_grad()
                outputs = net(inputs)
                log_probs = nn.functional.log_softmax(outputs, dim=1)

                # KLDivLoss expects input=log_probs, target=probs
                # Broadcast uniform_prob to batch size
                target_dist = uniform_prob.expand_as(log_probs)

                loss = -kl_loss_fn(log_probs, target_dist)  # Negative to do gradient ascent
                loss.backward()
                forget_optimizer.step()

        # Now fine-tune on retain set with cross-entropy
        net.train()
        epochs = 1
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for ep in range(epochs):
            for batch in retain_loader:
                if isinstance(batch, dict):
                    inputs = batch["image"] if "image" in batch else batch.get("inputs", None)
                    targets = batch.get("age_group", None)
                    if inputs is None or targets is None:
                        inputs, targets = batch["inputs"], batch["targets"]
                else:
                    inputs, targets = batch
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            scheduler.step()

        net.eval()
