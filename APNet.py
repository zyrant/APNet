from torch import nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torchvision.models as models
import torch

model = models.vgg16_bn(pretrained=True)
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

model2 = models.resnet50(pretrained=True)
model2_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class vgg_rgb(nn.Module):
    def __init__(self, pretrained=True):
        super(vgg_rgb, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # first model 224*24*64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),  # [:6]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, 3, 1, 1),  # second model 112*112*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # [6:13]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, 3, 1, 1),  # third model 56*56*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),  # [13:23]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, 3, 1, 1),  # forth model 28*28*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),  # [13:33]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, 3, 1, 1),  # fifth model 14*14*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),  # [33:43]
        )

        if pretrained:
            pretrained_vgg = model_zoo.load_url(model_urls['vgg16_bn'])
            model_dict = {}
            state_dict = self.state_dict()
            for k, v in pretrained_vgg.items():
                if k in state_dict:
                    model_dict[k] = v
                    # print(k, v)

        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def forward(self, rgb):
        A1 = self.features[:6](rgb)
        A2 = self.features[6:13](A1)
        A3 = self.features[13:23](A2)
        A4 = self.features[23:33](A3)
        A5 = self.features[33:43](A4)
        return A1, A2, A3, A4, A5
class vgg_thermal(nn.Module):
    def __init__(self, pretrained=True):
        super(vgg_thermal, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # first model 224*224*64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),  # [:6]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, 3, 1, 1),  # second model 112*112*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # [6:13]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, 3, 1, 1),  # third model 56*56*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),  # [13:23]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, 3, 1, 1),  # forth model 28*28*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),  # [13:33]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, 3, 1, 1),  # fifth model 14*14*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),  # [33:43]
        )

        if pretrained:
            pretrained_vgg = model_zoo.load_url(model_urls['vgg16_bn'])
            model_dict = {}
            state_dict = self.state_dict()
            for k, v in pretrained_vgg.items():
                if k in state_dict:
                    model_dict[k] = v
                    # print(k, v)

        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def forward(self, thermal):
        A1_d = self.features[:6](thermal)
        A2_d = self.features[6:13](A1_d)
        A3_d = self.features[13:23](A2_d)
        A4_d = self.features[23:33](A3_d)
        A5_d = self.features[33:43](A4_d)
        return A1_d, A2_d, A3_d, A4_d, A5_d

class Separable_conv(nn.Module):
    def __init__(self, inp, oup, ):
        super(Separable_conv, self).__init__()

        self.conv = nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, kernel_size=3, stride=1, padding=1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),
            # pw
            nn.Conv2d(inp, oup, kernel_size=1),
        )

    def forward(self, x):
        return self.conv(x)

# PGOS
class Mulit_Scale_dynamic_information_extraction2(nn.Module):
    def __init__(self, channel, r1, r2, p1, p2, reduction=4,):
        super(Mulit_Scale_dynamic_information_extraction2, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(channel, channel//2, (1, r1), 1, (0, p1)),
                                    nn.Conv2d(channel//2, channel//2, (r1, 1), 1, (p1, 0)),
                                    nn.BatchNorm2d(channel//2), nn.ReLU(inplace=True), )
        self.layer2 = nn.Sequential(nn.Conv2d(channel, channel//2, (1, r2), 1, (0, p2)),
                                    nn.Conv2d(channel//2, channel//2, (r2, 1), 1, (p2, 0)),
                                    nn.BatchNorm2d(channel//2), nn.ReLU(inplace=True), )
        # self.layer3 = nn.Sequential(nn.Conv2d(channel, channel // 2, (r1, 1), 1, (p1, 0)),)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, 1),
            nn.LayerNorm(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1),
            nn.Sigmoid()
        )
        self.nolocal = nn.Sequential(nn.Conv2d(channel, 1, 1),
                                     nn.Softmax(dim=2))

        self.conv1 = nn.Conv2d(channel * 4, channel, 1)
    def forward(self, x):
        x_1 = self.layer1(x)
        x_2 = self.layer2(x)
        x_cat = torch.cat((x_1, x_2), dim=1)
        nolocal = self.nolocal(x_cat)
        x_nolocal = x_cat + nolocal * x_cat
        b, c, _, _ = x_nolocal.size()
        y = self.avg_pool(x_nolocal).view(b, c, 1, 1)
        y = self.fc(y)
        out = torch.cat((x_nolocal*y.expand_as(x_nolocal), nolocal*x_cat, x_cat, x), dim=1)
        out = self.conv1(out)
        return out

class Mulit_Scale_dynamic_information_extraction4(nn.Module):
    def __init__(self, channel, r1, r2, p1, p2, reduction=4,):
        super(Mulit_Scale_dynamic_information_extraction4, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(channel, channel//4, (1, r1), 1, (0, p1)),
                                    nn.Conv2d(channel//4, channel//4, (r1, 1), 1, (p1, 0)),
                                    nn.BatchNorm2d(channel//4), nn.ReLU(inplace=True), )
        self.layer2 = nn.Sequential(nn.Conv2d(channel, channel//4, (1, r2), 1, (0, p2)),
                                    nn.Conv2d(channel//4, channel//4, (r2, 1), 1, (p2, 0)),
                                    nn.BatchNorm2d(channel//4), nn.ReLU(inplace=True), )
        self.layer3 = nn.Sequential(nn.Conv2d(channel, channel // 4, (1, r1), 1, (0, p1)),
                                    nn.Conv2d(channel // 4, channel // 4, (r1, 1), 1, (p1, 0)),
                                    nn.BatchNorm2d(channel // 4), nn.ReLU(inplace=True), )
        self.layer4 = nn.Sequential(nn.Conv2d(channel, channel // 4, (1, r2), 1, (0, p2)),
                                    nn.Conv2d(channel // 4, channel // 4, (r2, 1), 1, (p2, 0)),
                                    nn.BatchNorm2d(channel // 4), nn.ReLU(inplace=True), )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, 1),
            nn.LayerNorm(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1),
            nn.Sigmoid()
        )
        self.nolocal = nn.Sequential(nn.Conv2d(channel, 1, 1),
                                     nn.Softmax(dim=2))

        self.conv1 = nn.Conv2d(channel * 4, channel, 1)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, y):
        x_1 = self.layer1(x)
        x_2 = self.layer2(x)
        y = self.upsample(y)
        y_1 = self.layer3(y)
        y_2 = self.layer4(y)
        x_cat = torch.cat((x_1, y_1, x_2, y_2), dim=1)               # f
        nolocal = self.nolocal(x_cat)
        x_nolocal = x_cat + nolocal * x_cat                          # f + s
        b, c, _, _ = x_nolocal.size()
        z = self.avg_pool(x_nolocal).view(b, c, 1, 1)
        z = self.fc(z)
        out = torch.cat((x_nolocal * z.expand_as(x_nolocal), nolocal*x_cat, x_cat, x), dim=1)
        out = self.conv1(out)                                        # p
        return out

# PIEM
class Wfusion(nn.Module):
    def __init__(self,c):
        super(Wfusion, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.active = nn.Sigmoid()

    def forward(self, rgb, ti,fusion):
        cat_f = torch.cat((rgb, ti, fusion),dim=1)
        b, mc, _, _ = cat_f.size()
        act_f = self.active(torch.abs(cat_f))
        avg_f = self.avg_pool(act_f)
        weight = torch.split(avg_f,mc//3,dim=1)
        out = weight[0] * rgb + weight[1] * ti + weight[2] * (fusion)
        return out



class AINet(nn.Module):
    def __init__(self, ):
        super(AINet, self).__init__()
        self.rgb_pretrained = vgg_rgb()
        self.thermal_pretrained = vgg_thermal()

        #  res2net_decode层
        self.decode_layer5_r = Separable_conv(128, 64, ) # 64*224*224
        self.decode_layer4_r = Separable_conv(128, 64, )
        self.decode_layer3_r = Separable_conv(128, 64, )
        self.decode_layer2_r = Separable_conv(128, 64, )
        self.convrgb = nn.Conv2d(64, 1, 1)
        self.decode_layer1_r = Separable_conv(128, 64, )

        # upsample
        self.uplayer5 = nn.UpsamplingBilinear2d(scale_factor=16, )
        self.uplayer4 = nn.UpsamplingBilinear2d(scale_factor=8, )
        self.uplayer3 = nn.UpsamplingBilinear2d(scale_factor=4, )
        self.uplayer2 = nn.UpsamplingBilinear2d(scale_factor=2, )
        self.uplayer1 = nn.Sequential(
            Separable_conv(64, 64, ))

        self.changchannel5 = Separable_conv(512, 64, )
        self.changchannel4 = Separable_conv(512, 64, )
        self.changchannel3 = Separable_conv(256, 64, )
        self.changchannel2 = Separable_conv(128, 64, )
        self.changchannel1 = Separable_conv(64, 64, )

        # PGOS
        self.mscar5 = Mulit_Scale_dynamic_information_extraction2(64, 5, 3, 2, 1)
        self.mscar4 = Mulit_Scale_dynamic_information_extraction4(64, 7, 5, 3, 2)
        self.mscar3 = Mulit_Scale_dynamic_information_extraction4(64, 9, 7, 4, 3)
        self.mscar2 = Mulit_Scale_dynamic_information_extraction4(64, 11, 9, 5, 4)
        self.mscar1 = Mulit_Scale_dynamic_information_extraction4(64, 13, 11, 6, 5)



        # depth_decode层
        self.decode_layer5_d = Separable_conv(128, 64, )
        self.decode_layer4_d = Separable_conv(128, 64, )
        self.decode_layer3_d = Separable_conv(128, 64, )
        self.decode_layer2_d = Separable_conv(128, 64, )
        self.decode_layer1_d = Separable_conv(128, 64, )

        self.convd = nn.Conv2d(64, 1, 1)

        # depth_conv+upsample
        self.uplayer5_d = nn.UpsamplingBilinear2d(scale_factor=16, )
        self.uplayer4_d = nn.UpsamplingBilinear2d(scale_factor=8, )
        self.uplayer3_d = nn.UpsamplingBilinear2d(scale_factor=4, )
        self.uplayer2_d = nn.UpsamplingBilinear2d(scale_factor=2, )
        self.uplayer1_d = nn.Sequential(
            Separable_conv(64, 64, ))

        self.changchannel5_d = Separable_conv(512, 64, )
        self.changchannel4_d = Separable_conv(512, 64, )
        self.changchannel3_d = Separable_conv(256, 64, )
        self.changchannel2_d = Separable_conv(128, 64, )
        self.changchannel1_d = Separable_conv(64, 64, )

        self.mscad5 = Mulit_Scale_dynamic_information_extraction2(64, 5, 3, 2, 1)
        self.mscad4 = Mulit_Scale_dynamic_information_extraction4(64, 7, 5, 3, 2)
        self.mscad3 = Mulit_Scale_dynamic_information_extraction4(64, 9, 7, 4, 3)
        self.mscad2 = Mulit_Scale_dynamic_information_extraction4(64, 11, 9, 5, 4)
        self.mscad1 = Mulit_Scale_dynamic_information_extraction4(64, 13, 11, 6, 5)

        # PIEM
        self.final_out = Wfusion(1)

        self.sv4_f = nn.Conv2d(64, 1, 1)
        self.sv3_f = nn.Conv2d(64, 1, 1)
        self.sv2_f = nn.Conv2d(64, 1, 1)
        self.conv_f = nn.Conv2d(64, 1, 1)

    def forward(self, rgb, thermal):
        A1, A2, A3, A4, A5 = self.rgb_pretrained(rgb)
        A1_d, A2_d, A3_d, A4_d, A5_d = self.thermal_pretrained(thermal)

        # rgb
        F5 = self.changchannel5(A5)
        F4 = self.changchannel4(A4)
        F3 = self.changchannel3(A3)
        F2 = self.changchannel2(A2)
        F1 = self.changchannel1(A1)

        F5 = self.mscar5(F5)
        F4 = self.mscar4(F4, F5)
        F3 = self.mscar3(F3, F4)
        F2 = self.mscar2(F2, F3)
        F1 = self.mscar1(F1, F2)

        F5 = self.uplayer5(F5)
        F4 = self.uplayer4(F4)
        F3 = self.uplayer3(F3)
        F2 = self.uplayer2(F2)
        F1 = self.uplayer1(F1)

        f5 = torch.cat((F5, F4), dim=1)
        f4 = self.decode_layer5_r(f5)
        s4 = f4
        f4 = torch.cat((f4, F3), dim=1)
        f3 = self.decode_layer4_r(f4)
        s3 = f3
        f3 = torch.cat((F2, f3), dim=1)
        f2 = self.decode_layer3_r(f3)
        s2 = f2
        f2 = torch.cat((F1, f2), dim=1)
        f1 = self.decode_layer2_r(f2)
        f_r = self.convrgb(f1)

        # depth
        F5_d = self.changchannel5_d(A5_d)
        F4_d = self.changchannel4_d(A4_d)
        F3_d = self.changchannel3_d(A3_d)
        F2_d = self.changchannel2_d(A2_d)
        F1_d = self.changchannel1_d(A1_d)

        F5_d = self.mscad5(F5_d)
        F4_d = self.mscad4(F4_d, F5_d)
        F3_d = self.mscad3(F3_d, F4_d)
        F2_d = self.mscad2(F2_d, F3_d)
        F1_d = self.mscad1(F1_d, F2_d)

        F5_d = self.uplayer5_d(F5_d)
        F4_d = self.uplayer4_d(F4_d)
        F3_d = self.uplayer3_d(F3_d)
        F2_d = self.uplayer2_d(F2_d)
        F1_d = self.uplayer1_d(F1_d)

        f5_d = torch.cat((F5_d, F4_d), dim=1)
        f4_d = self.decode_layer5_d(f5_d)
        s4_d = f4_d
        f4_d = torch.cat((f4_d, F3_d), dim=1)
        f3_d = self.decode_layer4_d(f4_d)
        s3_d = f3_d
        f3_d = torch.cat((F2_d, f3_d), dim=1)
        f2_d = self.decode_layer3_d(f3_d)
        s2_d = f2_d
        f2_d = torch.cat((F1_d, f2_d), dim=1)
        f1_d = self.decode_layer2_d(f2_d)
        f_d = self.convd(f1_d)

        # supervision
        out4 = self.sv4_f(s4 + s4_d)
        out3 = self.sv3_f(s3 + s3_d)
        out2 = self.sv2_f(s2 + s2_d)


        out = self.conv_f(f1_d+f1)

        final_out = self.final_out(f_r, f_d, out)

        if self.training:
            return final_out, out, out4, out3, out2, f_r, f_d
        return final_out



if __name__=='__main__':

    # model = ghost_net()
    # model.eval()
    model = AINet()
    rgb = torch.randn(1, 3, 224, 224)
    depth = torch.randn(1, 3, 224, 224)
    out = model(rgb, depth)
    # print(out.shape)
    for i in out:
        print(i.shape)
    # from rgbd.FLOP import CalParams
    # CalParams(model,rgb,depth)