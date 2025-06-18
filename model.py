
import torch.nn as nn
import torchvision.models as models

class PneumoniaResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(PneumoniaResNet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        # 轉為單通道輸入 (1 channel)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 調整輸出層為2類
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
