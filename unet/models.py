import torch
import torch.nn as nn


class attenUnet(nn.Module):
    def __init__(self, nclasses=1):
        super(attenUnet, self).__init__()
        self.n_channels = 3
        self.n_classes = nclasses
        self.bilinear = False
        self.CNN512_main = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.CNN512_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.CNN256_main = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.CNN256_branch = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.CNN128_main = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.CNN128_branch = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.CNN64_main = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.CNN64_branch = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.CNN32_main = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.CNN32_branch = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.CNN32_main_up = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.CNN32_branch_up = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.CNN64_main_up = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.CNN64_branch_up = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.CNN128_main_up = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.CNN128_branch_up = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.CNN256_main_up = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.CNN256_branch_up = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.out = nn.Sequential(
            nn.Conv2d(32, nclasses, kernel_size=1)
        )

    def forward(self, img, mask):
        x1 = self.CNN512_main(img)
        y1 = self.CNN512_branch(mask)
        # x11 = torch.mul(x1, y1)
        x11 = torch.mul(x1, y1)+x1
        x12 = nn.MaxPool2d(kernel_size=2, stride=2)(x11)
        y1 = nn.MaxPool2d(kernel_size=2, stride=2)(y1)

        x2 = self.CNN256_main(x12)
        y2 = self.CNN256_branch(y1)
        # x21 = torch.mul(x2, y2)
        x21 = torch.mul(x2, y2)+x2
        x22 = nn.MaxPool2d(kernel_size=2, stride=2)(x21)
        y2 = nn.MaxPool2d(kernel_size=2, stride=2)(y2)

        x3 = self.CNN128_main(x22)
        y3 = self.CNN128_branch(y2)
        # x31 = torch.mul(x3, y3)
        x31 = torch.mul(x3, y3)+x3
        x32 = nn.MaxPool2d(kernel_size=2, stride=2)(x31)
        y3 = nn.MaxPool2d(kernel_size=2, stride=2)(y3)

        x4 = self.CNN64_main(x32)
        y4 = self.CNN64_branch(y3)
        # x41 = torch.mul(x4, y4)
        x41 = torch.mul(x4, y4)+x4
        x42 = nn.MaxPool2d(kernel_size=2, stride=2)(x41)
        y4 = nn.MaxPool2d(kernel_size=2, stride=2)(y4)

        x5 = self.CNN32_main(x42)
        y5 = self.CNN32_branch(y4)
        # x51 = torch.mul(x5, y5)
        x51 = torch.mul(x5, y5)+x5

        x6 = self.CNN32_main_up(x51)
        y6 = self.CNN32_branch_up(y5)
        # x61 = torch.mul(x6, y6) + x41
        x61 = torch.mul(x6, y6) +x6+ x41

        x7 = self.CNN64_main_up(x61)
        y7 = self.CNN64_branch_up(y6)
        # x71 = torch.mul(x7, y7) + x31
        x71 = torch.mul(x7, y7) +x7+ x31

        x8 = self.CNN128_main_up(x71)
        y8 = self.CNN128_branch_up(y7)
        # x81 = torch.mul(x8, y8) + x21
        x81 = torch.mul(x8, y8) +x8+ x21

        x9 = self.CNN256_main_up(x81)
        y9 = self.CNN256_branch_up(y8)
        # x91 = torch.mul(x9, y9) + x11
        x91 = torch.mul(x9, y9) +x9+ x11

        return self.out(x91)


if __name__ == '__main__':
    t = torch.rand((1, 3, 512, 512))
    mm = torch.rand((1, 1, 512, 512))
    m = attenUnet(1)
    res = m(t, mm)
    print(1)
