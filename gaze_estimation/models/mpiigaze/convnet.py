'''ResNet in PyTorch.

Reference:
https://github.com/kuangliu/pytorch-cifar
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import yacs.config


class Model(nn.Module):
    def __init__(self, config: yacs.config.CfgNode):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.fc1 = nn.Linear(192, 30)
        self.fc2 = nn.Linear(32, 2)
        # self.dropout = nn.Dropout(p=0.4)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # [128, 16, 36, 60]
        out1 = self.layer1(x)
        out12 = self.maxpool(out1)

        # [128, 32, 18, 30]
        out2 = self.layer2(out12)
        out23 = self.maxpool(out2)

        # [128, 32, 9, 15]
        out3 = self.layer3(out23)
        out34 = self.maxpool(out3)

        # [128, 32, 4, 7]
        out4 = self.layer4(out34)
        out = self.maxpool(out4)

        # out = out.reshape(out.size(0), -1)
        x = F.relu(self.fc1(out.view(out.size(0), -1)), inplace=True)
        x = torch.cat([x, y], dim=1)
        out = self.fc2(x)
        return out, [out1, out2, out3, out4]

