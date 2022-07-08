from .model import SiameseModel
from .loss import ContrastiveLoss
from .dataset import SiameseDataset
import torch
from .utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data_path = '/data/faces/training/'
test_data_path = '/data/faces/testing/'

def main():

    pass

if __name__ == '__main__':
    main()
