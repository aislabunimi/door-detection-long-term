import torch
from torch import nn

from doors_detection_long_term.doors_detector.models.gibson_goggle.gibson_goggle_definition import CompletionNet


class GibsonGoggle(nn.Module):
    def __init__(self):
        super(GibsonGoggle, self).__init__()

        comp = CompletionNet(norm=nn.BatchNorm2d, nf=64)
        comp = nn.DataParallel(comp)
        comp.load_state_dict(
        torch.load("gibson_goggle/unfiller_rgb.pth"))

        self.model = comp.module

    def forward(self, images, mask):
        return self.model(images, mask)