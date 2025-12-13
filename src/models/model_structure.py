import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class BasicBlock(nn.Module):
    """
    Базовый блок ResNet с residual connection.
    
    Args:
        in_channels (int): количество входных каналов.
        out_channels (int): количество выходных каналов.
        kernel_size (int): размер ядра свертки (должен быть нечётным).
        stride (int): шаг свертки.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int
        ):

        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve spatial dimensions.")
        
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Identity()
        if (stride != 1) or (in_channels != out_channels):
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out

class customResNet18(nn.Module):
    expansion = 2
    layer0_channels = 64
    def __init__(self, num_classes: int, zero_init_residual: bool = False):
        super().__init__()

        # Initial layers.
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=self.layer0_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.layer0_channels)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Main layers.
        order = 0
        self.layer1_0 = BasicBlock(
            in_channels = self.layer0_channels * self.expansion**order,
            out_channels = self.layer0_channels * self.expansion**order,
            kernel_size = 3,
            stride = 1
        )
        self.layer1_1 = BasicBlock(
            in_channels = self.layer0_channels * self.expansion**order,
            out_channels = self.layer0_channels * self.expansion**order,
            kernel_size = 3,
            stride = 1
        )

        order += 1
        self.layer2_0 = BasicBlock(
            in_channels = self.layer0_channels * self.expansion**(order-1),
            out_channels = self.layer0_channels * self.expansion**order,
            kernel_size = 3,
            stride = 2
        )
        self.layer2_1 = BasicBlock(
            in_channels = self.layer0_channels * self.expansion,
            out_channels = self.layer0_channels * self.expansion,
            kernel_size = 3,
            stride = 1
        )

        order += 1
        self.layer3_0 = BasicBlock(
            in_channels = self.layer0_channels * self.expansion**(order-1),
            out_channels = self.layer0_channels * self.expansion**order,
            kernel_size = 3,
            stride = 2
        )
        self.layer3_1 = BasicBlock(
            in_channels = self.layer0_channels * self.expansion**order,
            out_channels = self.layer0_channels * self.expansion**order,
            kernel_size = 3,
            stride = 1
        )
        
        # Отключаем 4-й слой, чтобы уместиться в 5 млн. параметров.
        '''
        order += 1
        self.layer4_0 = BasicBlock(
            in_channels = self.layer0_channels * self.expansion**(order-1),
            out_channels = self.layer0_channels * self.expansion**order,
            kernel_size = 3,
            stride = 1
        )
        self.layer4_1 = BasicBlock(
            in_channels = self.layer0_channels * self.expansion**order,
            out_channels = self.layer0_channels * self.expansion**order,
            kernel_size = 3,
            stride = 1
        )
        '''
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.layer0_channels * self.expansion**order, num_classes)

        # Init weights (optional, but recommended).
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-init batch norm in residual branches (improves convergence).
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print('input:', x.shape)
        x = self.conv1(x)
        print('input:', x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)
        print('layer0:', x.shape)
        
        x = self.layer1_0(x)
        x = self.layer1_1(x)
        print('layer1:', x.shape)
        
        x = self.layer2_0(x)
        x = self.layer2_1(x)
        print('layer2:', x.shape)

        x = self.layer3_0(x)
        x = self.layer3_1(x)
        print('layer3:', x.shape)

        # Отключаем 4-й слой, чтобы уместиться в 5 млн. параметров.
        '''
        x = self.layer4_0(x)
        x = self.layer4_1(x)
        print('layer4:', x.shape)
        '''

        x = self.avgpool(x)
        x = torch.flatten(x, 1) # безопасный аналог x.view(x.size(0), -1).
        x = self.fc(x)
        print('output:', x.shape)
        return x