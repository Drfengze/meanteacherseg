import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_c, r=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_c, in_c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_c // r, in_c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CombinedAttention(nn.Module):
    def __init__(self, in_c, r=16, kernel_size=7):
        super(CombinedAttention, self).__init__()
        self.channel_attention = ChannelAttention(in_c, r)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x_se = self.channel_attention(x)
        x_spatial = self.spatial_attention(x)
        return x_se * x_spatial

class conv_block(nn.Module):
    def __init__(self, in_c, out_c, attention_type=None):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.attention_type = attention_type
        self.channel_attention = ChannelAttention(out_c)
        self.spatial_attention = SpatialAttention()
        self.attention = CombinedAttention(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        if self.attention_type == 'se':
            x = self.channel_attention(x)
        elif self.attention_type == 'spatial':
            x = self.spatial_attention(x)
        elif self.attention_type == 'combined':
            x = self.attention(x)

        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c, attention_type=None):
        super().__init__()

        self.conv = conv_block(in_c, out_c, attention_type=attention_type)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c, attention_type=None):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c + out_c, out_c, attention_type=attention_type)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x

class build_unet(nn.Module):
    def __init__(self, bands, nclass, attention_type=None):
        super().__init__()
        self.bands = bands
        self.nclass = nclass
        """ Encoder """
        self.e1 = encoder_block(self.bands, 32, attention_type=attention_type)
        self.e2 = encoder_block(32, 64, attention_type=attention_type)
        self.e3 = encoder_block(64, 128, attention_type=attention_type)
        self.e4 = encoder_block(128, 256, attention_type=attention_type)
        self.e5 = encoder_block(256, 512, attention_type=attention_type)

        """ Bottleneck """
        self.b = conv_block(512, 1024, attention_type=attention_type)

        """ Decoder """
        self.d1 = decoder_block(1024, 512, attention_type=attention_type)
        self.d2 = decoder_block(512, 256, attention_type=attention_type)
        self.d3 = decoder_block(256, 128, attention_type=attention_type)
        self.d4 = decoder_block(128, 64, attention_type=attention_type)
        self.d5 = decoder_block(64, 32, attention_type=attention_type)

        """ Classifier """
        self.soft = nn.Softmax(dim=1)
        self.outputs = nn.Conv2d(32, self.nclass, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        s5, p5 = self.e5(p4)
        """ Bottleneck """
        b = self.b(p5)

        """ Decoder """
        d1 = self.d1(b, s5)
        d2 = self.d2(d1, s4)
        d3 = self.d3(d2, s3)
        d4 = self.d4(d3, s2)
        d5 = self.d5(d4, s1)
        """ Classifier """
        outputs = self.soft(self.outputs(d5))

        return outputs

