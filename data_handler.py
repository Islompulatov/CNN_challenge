import torch
from torchvision import datasets, transforms

def load_data(pth):
    transform = transforms.Compose([transforms.ToTensor(),  transforms.Normalize([0.5,], [0.5,])])


    trainset    = datasets.MNIST(pth, download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


    testset    = datasets.MNIST(pth, download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    return trainloader, testloader