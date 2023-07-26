import torch
from torch import nn

import torch.nn.functional as F

class AdaptiveNorm2d(nn.Module):
    def __init__(self, nchannel, momentum=0.05):
        super(AdaptiveNorm2d, self).__init__()
        self.nm = nn.BatchNorm2d(nchannel, momentum=momentum)
        self.w0 = nn.Parameter(torch.zeros(1))
        self.w1 = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.w0.repeat(x.size()) * self.nm(x) + self.w1.repeat(x.size()) * x


class CompletionNet(nn.Module):
    def __init__(self, norm=AdaptiveNorm2d, nf=64, skip_first_bn=False):
        super(CompletionNet, self).__init__()

        self.nf = nf
        alpha = 0.05
        if skip_first_bn:
            self.convs = nn.Sequential(
                nn.Conv2d(5, nf // 4, kernel_size=5, stride=1, padding=2),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf // 4, nf, kernel_size=5, stride=2, padding=2),
                norm(nf, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=2),
                norm(nf, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf, nf * 4, kernel_size=5, stride=2, padding=1),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, padding=1),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, padding=1),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, dilation=2, padding=2),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, dilation=4, padding=4),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, dilation=8, padding=8),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, dilation=16, padding=16),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, dilation=32, padding=32),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, padding=1),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, padding=1),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.ConvTranspose2d(nf * 4, nf, kernel_size=4, stride=2, padding=1),
                norm(nf, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),
                norm(nf, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.ConvTranspose2d(nf, nf // 4, kernel_size=4, stride=2, padding=1),
                norm(nf // 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf // 4, nf // 4, kernel_size=3, stride=1, padding=1),
                norm(nf // 4, momentum=alpha),
                nn.LeakyReLU(0.1),
                nn.Conv2d(nf // 4, 3, kernel_size=3, stride=1, padding=1),
            )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(5, nf // 4, kernel_size=5, stride=1, padding=2),
                norm(nf//4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf // 4, nf, kernel_size=5, stride=2, padding=2),
                norm(nf, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=2),
                norm(nf, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf, nf * 4, kernel_size=5, stride=2, padding=1),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, padding=1),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, padding=1),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, dilation=2, padding=2),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, dilation=4, padding=4),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, dilation=8, padding=8),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, dilation=16, padding=16),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, dilation=32, padding=32),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, padding=1),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, padding=1),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.ConvTranspose2d(nf * 4, nf, kernel_size=4, stride=2, padding=1),
                norm(nf, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),
                norm(nf, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.ConvTranspose2d(nf, nf // 4, kernel_size=4, stride=2, padding=1),
                norm(nf // 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf // 4, nf // 4, kernel_size=3, stride=1, padding=1),
                norm(nf // 4, momentum=alpha),
                nn.LeakyReLU(0.1),
                nn.Conv2d(nf // 4, 3, kernel_size=3, stride=1, padding=1),
            )

    def forward(self, x, mask):
        return F.tanh(self.convs(torch.cat([x, mask], 1)))