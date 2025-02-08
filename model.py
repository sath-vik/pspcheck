import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet import resnet101, resnet50

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, bias=False):
        super(ConvBlock, self).__init__()
        # Compute padding so that output dimensions are consistent
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1)) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

def upsample(input, size=None, scale_factor=None, align_corners=False):
    return F.interpolate(input, size=size, scale_factor=scale_factor, mode='bilinear', align_corners=align_corners)

class PyramidPooling(nn.Module):
    def __init__(self, in_channels):
        super(PyramidPooling, self).__init__()
        self.pooling_sizes = [1, 2, 3, 6]
        self.channels = in_channels // 4

        self.pool1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(self.pooling_sizes[0]),
            ConvBlock(in_channels, self.channels, kernel_size=1)
        )
        self.pool2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(self.pooling_sizes[1]),
            ConvBlock(in_channels, self.channels, kernel_size=1)
        )
        self.pool3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(self.pooling_sizes[2]),
            ConvBlock(in_channels, self.channels, kernel_size=1)
        )
        self.pool4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(self.pooling_sizes[3]),
            ConvBlock(in_channels, self.channels, kernel_size=1)
        )

    def forward(self, x):
        out1 = upsample(self.pool1(x), size=x.size()[-2:])
        out2 = upsample(self.pool2(x), size=x.size()[-2:])
        out3 = upsample(self.pool3(x), size=x.size()[-2:])
        out4 = upsample(self.pool4(x), size=x.size()[-2:])
        out = torch.cat([x, out1, out2, out3, out4], dim=1)
        return out

class PSPNet(nn.Module):
    def __init__(self, n_classes=21):
        """
        n_classes: number of segmentation classes.
        When using Cityscapes, pass the proper number (typically 19).
        """
        super(PSPNet, self).__init__()
        self.out_channels = 2048
        # Using resnet50 as backbone; modify here if you prefer resnet101.
        self.backbone = resnet50(pretrained=True)
        self.stem = nn.Sequential(*list(self.backbone.children())[:4])
        self.block1 = self.backbone.layer1
        self.block2 = self.backbone.layer2
        self.block3 = self.backbone.layer3
        self.block4 = self.backbone.layer4
        self.low_level_features_conv = ConvBlock(512, 64, kernel_size=3)

        self.depth = self.out_channels // 4
        self.pyramid_pooling = PyramidPooling(self.out_channels)
        self.decoder = nn.Sequential(
            ConvBlock(self.out_channels * 2, self.depth, kernel_size=3),
            nn.Dropout(0.1),
            nn.Conv2d(self.depth, n_classes, kernel_size=1)
        )
        self.aux = nn.Sequential(
            ConvBlock(self.out_channels // 2, self.depth // 2, kernel_size=3),
            nn.Dropout(0.1),
            nn.Conv2d(self.depth // 2, n_classes, kernel_size=1)
        )

        # Define the loss criteria.
        self.semantic_criterion = nn.CrossEntropyLoss(ignore_index=-100, weight=None).cuda()
        self.auxiliary_criterion = nn.CrossEntropyLoss(ignore_index=-100, weight=None).cuda()

    def forward(self, images, label=None):
        """
        Expect images as a dictionary.
        Typically, images['original_scale'] holds the input image.
        If multiple augmented views (e.g., flipped) are provided, they are averaged.
        """
        outs = []
        for key in images.keys():
            x = images[key]
            out = self.stem(x)
            out1 = self.block1(out)
            out2 = self.block2(out1)
            out3 = self.block3(out2)
            # Generate auxiliary output from an intermediate feature map.
            aux_out = self.aux(out3)
            aux_out = upsample(aux_out, size=images['original_scale'].size()[-2:], align_corners=True)
            out4 = self.block4(out3)

            out = self.pyramid_pooling(out4)
            out = self.decoder(out)
            out = upsample(out, size=x.size()[-2:], align_corners=True)
            out = upsample(out, size=images['original_scale'].size()[-2:], align_corners=True)

            if 'flip' in key:
                out = torch.flip(out, dims=[-1])
            outs.append(out)

        # Average the outputs if more than one augmentation is present.
        out = torch.stack(outs, dim=-1).mean(dim=-1)

        if label is not None:
            semantic_loss = self.semantic_criterion(out, label)
            aux_loss = self.auxiliary_criterion(aux_out, label)
            total_loss = semantic_loss + 0.4 * aux_loss
            return out, total_loss

        return out
