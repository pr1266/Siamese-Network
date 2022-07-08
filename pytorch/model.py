import torch
import torch.nn as nn
from utils import PrintLayer

class SiameseModel(nn.Module):

    def __init__(self):
        super(SiameseModel, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            PrintLayer(),
            #*******************
            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            PrintLayer(),
            #*******************
            nn.Conv2d(256, 384, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            PrintLayer(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(384, 1024),
            nn.ReLU(inplace=True),
            PrintLayer(),
            #*******************
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            PrintLayer(),
            #*******************
            nn.Linear(256,2),
            PrintLayer(),

        )

        def forward_(self, x):
            out = self.feature_extrctor(x)
            out = out.view(out.size()[0], -1)
            out = self.classifier(out)
            return out

        def forward(self, x1, x2):

            out1 = self.forward_(x1)
            out2 = self.forward_(x2)
            return out1, out2


def Test():

    model = SiameseModel()
    print(model)

if __name__ == '__main__':
    Test()