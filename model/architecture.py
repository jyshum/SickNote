"""
SickNoteResNet — ResNet18 backbone for binary cough classification.

Model outputs raw logits (no Sigmoid). Use BCEWithLogitsLoss for training.
Apply torch.sigmoid() at inference time only.

Pretrained on ImageNet. Input adapted from 3-channel to 1-channel.
"""
import torch
import torch.nn as nn
import torchvision.models as models


class SickNoteResNet(nn.Module):
    """ResNet18 with 1-channel input and binary output.

    conv1 weights initialized by averaging pretrained 3-channel weights.
    fc replaced with Linear(512, 1) for binary classification.
    """

    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights="IMAGENET1K_V1")

        # Adapt conv1: 3-channel → 1-channel
        old_conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight.copy_(old_conv1.weight.mean(dim=1, keepdim=True))

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # Replace classifier: 1000-class → binary
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        """Returns raw logits, shape (batch_size, 1)."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def backbone_params(self):
        """All parameters except the classifier head."""
        for name, param in self.named_parameters():
            if not name.startswith("fc."):
                yield param

    def head_params(self):
        """Classifier head parameters only."""
        return self.fc.parameters()
