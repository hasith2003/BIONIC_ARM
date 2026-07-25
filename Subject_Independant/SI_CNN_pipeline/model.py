import torch
import torch.nn as nn
import torch.nn.functional as F

class EMGSpectrogramCNN(nn.Module):
    def __init__(self, num_classes=10, f_bins=33, t_bins=12):
        super(EMGSpectrogramCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.2)
        
        flattened_h = f_bins // 2
        flattened_w = t_bins // 2
        self.fc_input_dim = 64 * flattened_h * flattened_w
        
        self.fc1 = nn.Linear(self.fc_input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ResidualBlock2D(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)

class EMGSpectrogramResNet(nn.Module):
    def __init__(self, num_classes=10, f_bins=33, t_bins=12):
        super(EMGSpectrogramResNet, self).__init__()
        self.conv_in = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(32)
        
        self.res1 = ResidualBlock2D(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv_mid = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn_mid = nn.BatchNorm2d(64)
        self.res2 = ResidualBlock2D(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        flat_h = f_bins // 4
        flat_w = t_bins // 4
        self.fc_input_dim = 64 * flat_h * flat_w
        
        self.fc1 = nn.Linear(self.fc_input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.pool1(self.res1(x))
        x = F.relu(self.bn_mid(self.conv_mid(x)))
        x = self.pool2(self.res2(x))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
