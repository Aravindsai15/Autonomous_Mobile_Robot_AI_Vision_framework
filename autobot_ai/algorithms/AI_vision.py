
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import os

def remap_labels(mask):
    """
    Remap original dataset labels to:
    - 0: road
    - 1: sidewalk
    - 2: building
    - 3: vegetation
    - 4: sky
    - 255: ignore all other classes
    """
    new_mask = torch.full_like(mask, 255)
    new_mask[mask == 7] = 0    # road
    new_mask[mask == 8] = 1    # sidewalk
    new_mask[mask == 11] = 2   # building
    new_mask[mask == 21] = 3   # vegetation
    new_mask[mask == 23] = 4   # sky
    return new_mask




# Depthwise Separable Conv Block
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel, stride, padding, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = torch.clamp(F.relu(self.bn1(self.depthwise(x))), max=6)
        x = torch.clamp(F.relu(self.bn2(self.pointwise(x))), max=6)
        return x

# Simple Channel Attention Block (replacing SE)
class SimpleChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class LearningToDownsample(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, 32, 3, stride=2, padding=1)
        self.conv2 = DepthwiseSeparableConv(32, 48, 3, stride=2, padding=1)
        self.conv3 = DepthwiseSeparableConv(48, out_channels, 3, stride=2, padding=1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        return x

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_ratio=6):
        super().__init__()
        hidden_channels = in_channels * expansion_ratio
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)

        # Replace conv2 with depthwise conv + pointwise conv
        self.depthwise = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=stride, padding=1, groups=hidden_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.pointwise = nn.Conv2d(hidden_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.attention = SimpleChannelAttention(out_channels, reduction=8)  # simpler attention

        self.stride = stride
        self.use_res_connect = (stride == 1 and in_channels == out_channels)

    def forward(self, x):
        identity = x
        x = torch.clamp(F.relu(self.bn1(self.conv1(x))), max=6)
        x = torch.clamp(F.relu(self.bn2(self.depthwise(x))), max=6)
        x = self.bn3(self.pointwise(x))

        if self.use_res_connect:
            x = self.attention(x)
            return x + identity
        return x

class GlobalFeatureExtractor(nn.Module):
    def __init__(self, in_channels=64, out_channels=128):
        super().__init__()
        layers = [
            Bottleneck(in_channels, 64, stride=2),
            Bottleneck(64, 64, stride=1),
            Bottleneck(64, 128, stride=2),
            Bottleneck(128, 128, stride=1),
            Bottleneck(128, out_channels, stride=1)
        ]
        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.blocks(x)

class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x1, x2):
        x2 = F.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x1, x2], dim=1)
        x = torch.clamp(F.relu(self.bn1(self.conv1(x))), max=6)
        x = self.dropout(x)
        x = self.conv2(x)
        return x

class FastSCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.lds = LearningToDownsample()
        self.gfe = GlobalFeatureExtractor()
        self.ffm = FeatureFusionModule(in_channels=64+128)

        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, 1)
        )

        self.aux_classifier = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, 1)
        )

    def forward(self, x):
        input_size = x.shape[2:]

        x1 = self.lds(x)
        x2 = self.gfe(x1)

        x = self.ffm(x1, x2)
        main_out = self.classifier(x)
        main_out = F.interpolate(main_out, size=input_size, mode='bilinear', align_corners=True)

        aux_out = self.aux_classifier(x2)
        aux_out = F.interpolate(aux_out, size=input_size, mode='bilinear', align_corners=True)

        return main_out, aux_out




# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import CosineAnnealingLR
# from torchvision import transforms
# import numpy as np
# from tqdm import tqdm
# from sklearn.metrics import confusion_matrix
# from PIL import Image
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# import matplotlib.patches as mpatches
# import os

# # ------------------------------
# # 2. Fast-SCNN Model Definition
# # ------------------------------
# class LearningToDownsample(nn.Module):
#     def __init__(self, in_channels=3, out_channels=64):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 48, 3, stride=2, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(48)
#         self.conv3 = nn.Conv2d(48, out_channels, 3, stride=2, padding=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_channels)
#         self.dropout = nn.Dropout2d(0.1)

#     def forward(self, x):
#         x = F.relu6(self.bn1(self.conv1(x)))
#         x = self.dropout(x)
#         x = F.relu6(self.bn2(self.conv2(x)))
#         x = self.dropout(x)
#         x = F.relu6(self.bn3(self.conv3(x)))
#         return x

# class Bottleneck(nn.Module):
#     def __init__(self, in_channels, out_channels, stride, expansion_ratio=6):
#         super().__init__()
#         hidden_channels = in_channels * expansion_ratio
#         self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
#         self.bn1 = nn.BatchNorm2d(hidden_channels)
#         self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, 
#                               stride=stride, padding=1, 
#                               groups=hidden_channels, bias=False)
#         self.bn2 = nn.BatchNorm2d(hidden_channels)
#         self.conv3 = nn.Conv2d(hidden_channels, out_channels, 1, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_channels)
# #         self.se = nn.Sequential(
# #                 nn.AdaptiveAvgPool2d(1),
# #                 nn.Conv2d(out_channels, out_channels // 4, 1),  # Use out_channels here
# #                 nn.ReLU(),
# #                 nn.Conv2d(out_channels // 4, out_channels, 1),
# #                 nn.Sigmoid()
# #             )
        
        
#         self.se = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(out_channels, out_channels//4, 1),
#             nn.ReLU(),
#             nn.Conv2d(out_channels//4, out_channels, 1),
#             nn.Sigmoid()
#         )
        
        
#         self.stride = stride
#         self.use_res_connect = (stride == 1 and in_channels == out_channels)

#     def forward(self, x):
#         identity = x
#         x = F.relu6(self.bn1(self.conv1(x)))
#         x = F.relu6(self.bn2(self.conv2(x)))
#         x = self.bn3(self.conv3(x))
        
#         if self.use_res_connect:
#             se_weight = self.se(x)
#             x = x * se_weight
#             return x + identity
#         return x

# class GlobalFeatureExtractor(nn.Module):
#     def __init__(self, in_channels=64, out_channels=128):
#         super().__init__()
#         layers = [
#             Bottleneck(in_channels, 64, stride=2),
#             Bottleneck(64, 64, stride=1),
#             Bottleneck(64, 128, stride=2),
#             Bottleneck(128, 128, stride=1),
#             Bottleneck(128, out_channels, stride=1)
#         ]
#         self.blocks = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.blocks(x)

# class FeatureFusionModule(nn.Module):
#     def __init__(self, in_channels, out_channels=256):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
        
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()
#         )
        
#         self.dropout = nn.Dropout2d(0.2)

#     def forward(self, x1, x2):
#         x2 = F.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=True)
#         x = torch.cat([x1, x2], dim=1)
#         x = F.relu6(self.bn1(self.conv1(x)))
#         x = self.dropout(x)
#         x = self.conv2(x)
#         return x

# class FastSCNN(nn.Module):
#     def __init__(self, num_classes=4):
#         super().__init__()
#         # Initialize weights properly
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#         self.lds = LearningToDownsample()
#         self.gfe = GlobalFeatureExtractor()
#         self.ffm = FeatureFusionModule(in_channels=64+128)
        
#         self.classifier = nn.Sequential(
#             nn.Conv2d(256, 256, 3, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Dropout2d(0.1),
#             nn.Conv2d(256, num_classes, 1)
#         )
        
#         self.aux_classifier = nn.Sequential(
#             nn.Conv2d(128, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, num_classes, 1)
#         )

#     def forward(self, x):
#         input_size = x.shape[2:]
        
#         x1 = self.lds(x)
#         x2 = self.gfe(x1)
        
#         # Main branch
#         x = self.ffm(x1, x2)
#         main_out = self.classifier(x)
#         main_out = F.interpolate(main_out, size=input_size, mode='bilinear', align_corners=True)
        
#         # Aux branch
#         aux_out = self.aux_classifier(x2)
#         aux_out = F.interpolate(aux_out, size=input_size, mode='bilinear', align_corners=True)
        
#         return main_out, aux_out
