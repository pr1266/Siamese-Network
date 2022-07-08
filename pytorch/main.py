from model import SiameseModel
from loss import ContrastiveLoss
from dataset import SiameseDataset
import torch
import torch.optim as optim
from utils import *
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data_path = './data/faces/training/'
test_data_path = './data/faces/testing/'

def main():

    folder_dataset = datasets.ImageFolder(root=train_data_path)
    transformation = transforms.Compose([transforms.Resize((100,100)), transforms.ToTensor()])
    siamese_dataset = SiameseDataset(path=folder_dataset,transform=transformation)
    train_dataloader = DataLoader(siamese_dataset, shuffle=True, num_workers=8, batch_size=64)
    net = SiameseModel().to(device=device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr = 0.0005 )
if __name__ == '__main__':
    main()
