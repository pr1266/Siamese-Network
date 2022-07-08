from model import SiameseModel
from loss import ContrastiveLoss
from dataset import SiameseDataset
import torch
import torch.optim as optim
from utils import *
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data_path = './data/faces/training/'
test_data_path = './data/faces/testing/'
epochs = 100

def main():

    folder_dataset = datasets.ImageFolder(root=train_data_path)
    transformation = transforms.Compose([transforms.Resize((100,100)), transforms.ToTensor()])
    siamese_dataset = SiameseDataset(path=folder_dataset,transform=transformation)
    train_dataloader = DataLoader(siamese_dataset, shuffle=True, num_workers=8, batch_size=64)
    net = SiameseModel().to(device=device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr = 0.0005 )
    
    #! Train Phase:
    counter = []
    loss_history = [] 
    iteration_number= 0

    for epoch in range(epochs):
        for i, (img0, img1, label) in enumerate(train_dataloader, 0):
            img0, img1, label = img0.to(device=device), img1.to(device=device), label.to(device=device)
            optimizer.zero_grad()
            output1, output2 = net(img0, img1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
            if i % 10 == 0 :
                print(f"Epoch number {epoch}\n Current loss {loss_contrastive.item()}\n")
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())

    show_plot(counter, loss_history)

    folder_dataset_test = datasets.ImageFolder(root=test_data_path)
    siamese_dataset = SiameseDataset(path=folder_dataset_test, transform=transformation)
    test_dataloader = DataLoader(siamese_dataset, num_workers=2, batch_size=1, shuffle=True)
    dataiter = iter(test_dataloader)
    x0, _, _ = next(dataiter)

    for i in range(10):
        _, x1, label2 = next(dataiter)
        concatenated = torch.cat((x0, x1), 0)
        output1, output2 = net(x0.cuda(), x1.cuda())
        euclidean_distance = F.pairwise_distance(output1, output2)
        imshow(torchvision.utils.make_grid(concatenated), f'Dissimilarity: {euclidean_distance.item():.2f}')
if __name__ == '__main__':
    main()
