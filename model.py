import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import resnet101, resnet50

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, bias=False):
        super(ConvBlock, self).__init__()
        # Proper padding calculation for same-size output
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1)) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                     stride=stride, padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

def upsample(input, size=None, scale_factor=None, align_corners=False):
    return F.interpolate(input, size=size, scale_factor=scale_factor, 
                        mode='bilinear', align_corners=align_corners)

class PyramidPooling(nn.Module):
    def __init__(self, in_channels):
        super(PyramidPooling, self).__init__()
        self.pooling_sizes = [1, 2, 3, 6]
        self.channels = in_channels // 4  # 2048//4=512 per pyramid level

        self.pools = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                ConvBlock(in_channels, self.channels, kernel_size=1)
            ) for size in self.pooling_sizes
        ])

    def forward(self, x):
        h, w = x.size()[2:]
        pyramid_features = [x]
        for pool in self.pools:
            pooled = pool(x)
            upsampled = F.interpolate(pooled, size=(h,w), mode='bilinear', align_corners=True)
            pyramid_features.append(upsampled)
        return torch.cat(pyramid_features, dim=1)  # 2048 + 4*512 = 4096

class PSPNet(nn.Module):
    def __init__(self, n_classes=19):
        super(PSPNet, self).__init__()
        self.out_channels = 2048
        self.backbone = resnet50(pretrained=True)
        
        # Backbone decomposition
        self.stem = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool
        )
        self.block1 = self.backbone.layer1
        self.block2 = self.backbone.layer2
        self.block3 = self.backbone.layer3
        self.block4 = self.backbone.layer4

        # Class-balancing weights (approximate Cityscapes distribution)
        class_weights = torch.tensor([
            0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969, 0.9754,
            1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
            1.0873, 1.0955, 1.0865, 1.1529, 1.0507
        ]).cuda()

        self.low_level_features_conv = ConvBlock(256, 64, kernel_size=1)  # Correct channel from layer1
        
        self.pyramid_pooling = PyramidPooling(self.out_channels)
        self.decoder = nn.Sequential(
            ConvBlock(4096, 512, kernel_size=3),  # 4096 from pyramid
            nn.Dropout(0.1),
            nn.Conv2d(512, n_classes, kernel_size=1)
        )

        # Enhanced auxiliary branch
        self.aux = nn.Sequential(
            ConvBlock(1024, 256, kernel_size=3),  # From layer3 output
            ConvBlock(256, 128, kernel_size=3),
            nn.Dropout(0.1),
            nn.Conv2d(128, n_classes, kernel_size=1)
        )

        # Loss functions with class weights
        self.semantic_criterion = nn.CrossEntropyLoss(
            ignore_index=255,  # Matches preprocess.py mapping
            weight=class_weights
        ).cuda()
        
        self.auxiliary_criterion = nn.CrossEntropyLoss(
            ignore_index=255,
            weight=class_weights
        ).cuda()

    def forward(self, images, labels=None):
        outs = []
        aux_outs = []
        
        for key in images.keys():
            x = images[key]
            # Backbone features
            x = self.stem(x)
            x = self.block1(x)   # 256 channels
            x = self.block2(x)   # 512 channels
            x_aux = self.block3(x)  # 1024 channels
            x = self.block4(x_aux)  # 2048 channels

            # Auxiliary output
            aux_out = self.aux(x_aux)
            aux_out = upsample(aux_out, size=images['original_scale'].shape[-2:], align_corners=False)
            aux_outs.append(aux_out)

            # Main branch
            x = self.pyramid_pooling(x)
            x = self.decoder(x)
            x = upsample(x, size=images['original_scale'].shape[-2:], align_corners=False)
            
            if 'flip' in key:
                x = torch.flip(x, dims=[-1])
            outs.append(x)

        # Average outputs
        main_out = torch.stack(outs, dim=-1).mean(dim=-1)
        aux_out = torch.stack(aux_outs, dim=-1).mean(dim=-1)

        if labels is not None:
            main_loss = self.semantic_criterion(main_out, labels)
            aux_loss = self.auxiliary_criterion(aux_out, labels)
            return main_out, main_loss + 0.4 * aux_loss
        
        return main_out
