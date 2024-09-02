import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .pvtv2 import *

class ConvUnit(nn.Module):
    def __init__(self, features):
        super(ConvUnit, self).__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x

class FeatureFusionBlock(nn.Module):
    def __init__(self, features):
        super(FeatureFusionBlock, self).__init__()
        self.resConfUnit1 = ConvUnit(features)
        self.resConfUnit2 = ConvUnit(features)

    def forward(self, *xs):
        output = xs[0]

        if len(xs) == 2:
            output = F.interpolate(output, size=xs[1].size()[2:], mode="bilinear", align_corners=True)
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        return output

class Prototype_Guide_Fusion(nn.Module):
    def __init__(self, channel):
        super(Prototype_Guide_Fusion, self).__init__()
        self.prob = nn.Sigmoid()
        self.fusion = FeatureFusionBlock(channel)

    def calculate_prototype(self, f, m):
        m = F.interpolate(m, f.size()[2:], mode='bilinear', align_corners= True)
        m = m>0.5
        masked_feature = f * m.float()
        prototype = masked_feature.sum(dim=(2,3)) / (m.sum(dim=(2,3))+ 1e-8)

        return prototype

    def measure_similarity(self, f, p):
        sim = F.cosine_similarity(f, p[..., None, None], dim=1)
        sim = sim[:,None,...]

        return sim

    def forward(self, high_level, low_level, predict):
        prob_map = self.prob(predict)
        foreground_prototype = self.calculate_prototype(high_level, prob_map)
        background_prototype = self.calculate_prototype(high_level, 1-prob_map)

        low_level = self.fusion(high_level, low_level)

        low_level_fore = self.measure_similarity(low_level, foreground_prototype) * low_level

        low_level_back = self.measure_similarity(low_level, background_prototype) * low_level


        return low_level_fore, low_level_back


class Context_Exploration_Block(nn.Module):
    def __init__(self, input_channels):
        super(Context_Exploration_Block, self).__init__()
        self.input_channels = input_channels
        self.channels_single = int(input_channels / 4)

        self.p1_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p1 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p1_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p2 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p3 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 5, 1, 2),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p4 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 7, 1, 3),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.fusion = nn.Sequential(nn.Conv2d(self.input_channels, self.input_channels, 1, 1, 0),
                                    nn.BatchNorm2d(self.input_channels), nn.ReLU())

    def forward(self, x):
        p1_input = self.p1_channel_reduction(x)
        p1 = self.p1(p1_input)
        p1_dc = self.p1_dc(p1)

        p2_input = self.p2_channel_reduction(x) + p1_dc
        p2 = self.p2(p2_input)
        p2_dc = self.p2_dc(p2)

        p3_input = self.p3_channel_reduction(x) + p2_dc
        p3 = self.p3(p3_input)
        p3_dc = self.p3_dc(p3)

        p4_input = self.p4_channel_reduction(x) + p3_dc
        p4 = self.p4(p4_input)
        p4_dc = self.p4_dc(p4)

        ce = self.fusion(torch.cat((p1_dc, p2_dc, p3_dc, p4_dc), 1))

        return ce


class Building_Mining_Module(nn.Module):
    def __init__(self, channel,num_classes):
        super(Building_Mining_Module, self).__init__()
        self.PGF = Prototype_Guide_Fusion(channel)
        self.output_map = nn.Conv2d(channel, num_classes, 7, 1, 3)

        self.fp = Context_Exploration_Block(channel)
        self.fn = Context_Exploration_Block(channel)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(channel)
        self.relu2 = nn.ReLU()

    def forward(self, x, y, in_map):
        in_map = in_map[:, 1, :, :].unsqueeze(1)

        f_feature, b_feature = self.PGF(y, x, in_map)

        fp = self.fp(f_feature)
        fn = self.fn(b_feature)

        refine1 = self.alpha * fp
        refine1 = self.bn1(refine1)
        refine1 = self.relu1(refine1)

        refine2 = refine1 + (self.beta * fn)
        refine2 = self.bn2(refine2)
        refine2 = self.relu2(refine2)

        output_map = self.output_map(refine2)

        return refine2, output_map


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Model_3(nn.Module):
    def __init__(self, channel=32, num_classes=1):
        super(Model_3, self).__init__()
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_model/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.cr4 = nn.Sequential(nn.Conv2d(512, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.ReLU())
        self.cr3 = nn.Sequential(nn.Conv2d(320, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.ReLU())
        self.cr2 = nn.Sequential(nn.Conv2d(128, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.ReLU())
        self.cr1 = nn.Sequential(nn.Conv2d(64, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.ReLU())

        self.ca_1 = ChannelAttention(channel)
        self.sa_1 = SpatialAttention()

        self.ca_2 = ChannelAttention(channel)
        self.sa_2 = SpatialAttention()

        self.ca_3 = ChannelAttention(channel)
        self.sa_3 = SpatialAttention()

        self.ca_4 = ChannelAttention(channel)
        self.sa_4 = SpatialAttention()

        self.out_4 = nn.Sequential(nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.ReLU(),
                                   nn.Conv2d(channel, num_classes, 3, 1, 1))

        self.bmm3 = Building_Mining_Module(channel, num_classes)
        self.bmm2 = Building_Mining_Module(channel, num_classes)
        self.bmm1 = Building_Mining_Module(channel, num_classes)

    def forward(self, x):

        layer1, layer2, layer3, layer4 = self.backbone(x)

        layer4 = self.cr4(layer4)
        layer3 = self.cr3(layer3)
        layer2 = self.cr2(layer2)
        layer1 = self.cr1(layer1)

        layer4 = self.ca_4(layer4) * layer4  # channel attention
        layer4 = self.sa_4(layer4) * layer4  # spatial attention

        layer3 = self.ca_3(layer3) * layer3  # channel attention
        layer3 = self.sa_3(layer3) * layer3  # spatial attention

        layer2 = self.ca_2(layer2) * layer2  # channel attention
        layer2 = self.sa_2(layer2) * layer2  # spatial attention

        layer1 = self.ca_1(layer1) * layer1  # channel attention
        layer1 = self.sa_1(layer1) * layer1  # spatial attention

        predict4 = self.out_4(layer4)

        fusion, predict3 = self.bmm3(layer3, layer4, predict4)
        fusion, predict2 = self.bmm2(layer2, fusion, predict3)
        fusion, predict1 = self.bmm1(layer1, fusion, predict2)

        predict4 = F.interpolate(predict4, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict3 = F.interpolate(predict3, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict2 = F.interpolate(predict2, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict1 = F.interpolate(predict1, size=x.size()[2:], mode='bilinear', align_corners=True)

        return predict4, predict3, predict2, predict1


if __name__ == '__main__':
    x = torch.rand(4,3,512,512).cuda()
    model = Model_3(32,2).cuda()
    out = model(x)
    for i in out:
        print(i.shape)
