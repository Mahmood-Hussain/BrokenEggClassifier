import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class EggClassifierV2(nn.Module):
    def __init__(self, in_channels, num_classes=3):
        super().__init__()

        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x
    
# test the model
if __name__ == '__main__':
    import torch
    model = EggClassifierV2(3)
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    # print(model)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)
