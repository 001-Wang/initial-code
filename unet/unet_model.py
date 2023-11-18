""" Full assembly of the parts to form the complete network """


# from unet_parts import *
import sys
sys.path.append('/workspace/wzx/attenUnet2/unet/')
from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x,y):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class attenUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, bilinear=False):
        super(attenUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.inc_mask=DoubleConv(1, 64)
        self.down1_mask = Down(64, 128)
        self.down2_mask = Down(128, 256)
        self.down3_mask = Down(256, 512)

        self.down4_mask = Down(512, 1024 // factor)
        self.up1_mask = Up(1024, 512 // factor, bilinear)
        self.up2_mask = Up(512, 256 // factor, bilinear)
        self.up3_mask = Up(256, 128 // factor, bilinear)
        # self.up4_mask = Up(128, 64, bilinear)


        self.outc = OutConv(64, n_classes)

    def forward(self, x,y):

        y1 = self.inc_mask(y)
        y2 = self.down1_mask(y1)
        y3 = self.down2_mask(y2)
        y4 = self.down3_mask(y3)
        y5 = self.down4_mask(y4)
        y44 = self.up1_mask(y5, y4)
        y33 = self.up2_mask(y44, y3)
        y22 = self.up3_mask(y33, y2)
        # y11 = self.up4_mask(y22, y1)


        x1 = self.inc(x)
        x2 = self.down1(x1*y1+x1)
        x3 = self.down2(x2*y2+x2)
        x4 = self.down3(x3*y3+x3)
        x5 = self.down4(x4*y4+x4)
        x = self.up1(x5*y5+x5, x4)
        x = self.up2(x*y44+x, x3)
        x = self.up3(x*y33+x, x2)
        x = self.up4(x*y22+x, x1)

        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)

        logits = self.outc(x)
        return logits



class UNet_at_without_d(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, bilinear=False):
        super(UNet_at_without_d, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # self.inc_mask=DoubleConv(1, 64)
        # self.down1_mask = Down(64, 128)
        # self.down2_mask = Down(128, 256)
        # self.down3_mask = Down(256, 512)
        #
        # self.down4_mask = Down(512, 1024 // factor)
        # self.up1_mask = Up(1024, 512 // factor, bilinear)
        # self.up2_mask = Up(512, 256 // factor, bilinear)
        # self.up3_mask = Up(256, 128 // factor, bilinear)
        # self.up4_mask = Up(128, 64, bilinear)


        self.outc = OutConv(64, n_classes)

    def forward(self, x,y):

        # y1 = self.inc_mask(y)
        # y2 = self.down1_mask(y1)
        # y3 = self.down2_mask(y2)
        # y4 = self.down3_mask(y3)
        # y5 = self.down4_mask(y4)
        # y44 = self.up1_mask(y5, y4)
        # y33 = self.up2_mask(y44, y3)
        # y22 = self.up3_mask(y33, y2)
        # y11 = self.up4_mask(y22, y1)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # x1 = self.inc(x)
        # x2 = self.down1(x1*y1+x1)
        # x3 = self.down2(x2*y2+x2)
        # x4 = self.down3(x3*y3+x3)
        # x5 = self.down4(x4*y4+x4)
        # x = self.up1(x5*y5+x5, x4)
        # x = self.up2(x*y44+x, x3)
        # x = self.up3(x*y33+x, x2)
        # x = self.up4(x*y22+x, x1)

        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)

        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    t = torch.rand((1, 3, 512, 512))
    mm = torch.rand((1, 1, 512, 512))
    # m = attenUNet(3,2)
    # m = UNet(3,2)
    m = UNet_at_without_d(3,2)
    res = m(t, mm)
    print(1)
