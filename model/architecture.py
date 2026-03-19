import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSepConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, stride=stride,
            padding=1, groups=in_channels
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.relu(x)


class AudioCNN(nn.Module):
    def __init__(self, num_classes=35):
        super().__init__()

        self.input_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.block1 = DepthwiseSepConv(16, 32, stride=2)
        self.block2 = DepthwiseSepConv(32, 64, stride=2)
        self.block3 = DepthwiseSepConv(64, 128, stride=2)

        self.dropout = nn.Dropout(0.3)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.input_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = AudioCNN(num_classes=35)

    dummy_input = torch.randn(1, 1, 64, 101)
    output = model(dummy_input)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"model architecture:")
    print(model)
    print(f"\ninput shape:  {dummy_input.shape}")
    print(f"output shape: {output.shape}")
    print(f"total params: {total_params:,}")
    print(f"trainable:    {trainable_params:,}")
    print(f"target:       under 500,000 params")

    if total_params < 500000:
        print(f"PASS — model fits FPGA constraints")
    else:
        print(f"WARNING — model too large for edge deployment")