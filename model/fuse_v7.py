import torch
import torch.nn as nn
import torch.nn.functional as F

def updown(x, size, mode='bilinear'):
    out = F.interpolate(x, size=size, mode=mode, align_corners=True)
    return out

class Pre(nn.Module):
    def __init__(self, channels):
        super(Pre, self).__init__()

        self.preconv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16, out_channels=channels, kernel_size=3, padding=1, stride=1, bias=True)
        )
    def forward(self, pan, ms_channel):
        # print(pan.shape, '\n', ms_channel.shape)
        concat = torch.cat((pan, ms_channel), 1)
        preconv = self.preconv(concat)

        return preconv

class fusion(nn.Module):
    def __init__(self, channels):
        super(fusion, self).__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, stride=1, bias=True),
        )

    def forward(self, x):
        fusion = self.fusion(x)
        out = fusion + x
        return out


class fuse(nn.Module):
    def __init__(self, channels, num_of_layers):
        super(fuse, self).__init__()
        self.up4_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=5, out_channels=4, kernel_size=4, padding=1, stride=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1, stride=1,bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, padding=1, stride=1,bias=True),
        )

        self.up4_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=5, out_channels=4, kernel_size=4, padding=1, stride=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, padding=1, stride=1, bias=True)
        )

        self.up8_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=9, out_channels=8, kernel_size=4, padding=1, stride=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1, stride=1, bias=True)
        )

        self.up8_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=9, out_channels=8, kernel_size=4, padding=1, stride=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1, stride=1, bias=True)
        )

        self.pre_list4 = nn.ModuleList([Pre(channels=channels) for i in range(4)])
        self.pre_list8 = nn.ModuleList([Pre(channels=channels) for i in range(8)])

        self.fusion_list = nn.ModuleList([fusion(channels=channels) for i in range(num_of_layers)])

        # recon
        self.recon4 = nn.Sequential(
            nn.Conv2d(in_channels=channels*4, out_channels=channels, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=channels, out_channels=4, kernel_size=3, padding=1, stride=1, bias=True),
        )

        self.recon8 = nn.Sequential(
            nn.Conv2d(in_channels=channels * 8, out_channels=channels, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=channels, out_channels=8, kernel_size=3, padding=1, stride=1, bias=True),
        )

    def forward(self, pan, lr):
        _, N, H, W = lr.shape
        pan_4 = updown(pan, (H, W))
        pan_2 = updown(pan, (H*2, W*2))
        concat1 = torch.cat((lr, pan_4), 1)
        if N == 4:
            lr_2 = self.up4_1(concat1)
            concat2 = torch.cat((lr_2, pan_2), 1)
            lr_u = self.up4_2(concat2)
        else:
            lr_2 = self.up8_1(concat1)
            concat2 = torch.cat((lr_2, pan_2), 1)
            lr_u = self.up8_2(concat2)
        pre = []
        if N == 4:
            for i in range(N):
                temp = self.pre_list4[i](pan, lr_u[:, i, None, :])
                temp1 = self.fusion_list[0](temp)
                pre.append(temp1)
            for i in range(1, len(self.fusion_list)):
                for j in range(N):
                    pre[j] = self.fusion_list[i](pre[j])
            concat = torch.cat(pre, dim=1)
            output = self.recon4(concat)
        else:
            for i in range(N):
                temp = self.pre_list8[i](pan, lr_u[:, i, None, :])
                temp1 = self.fusion_list[0](temp)
                pre.append(temp1)
            for i in range(1, len(self.fusion_list)):
                for j in range(N):
                    pre[j] = self.fusion_list[i](pre[j])
            concat = torch.cat(pre, dim=1)
            output = self.recon8(concat)
        output = torch.sigmoid(output)
        return lr_2, lr_u, output
