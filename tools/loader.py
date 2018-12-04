import torchvision.transforms as transforms
import torchvision
import torch

transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class Loader():
  def __init__(self, dataset, batch, path, download):
    self.trainset = ''
    self.testset = ''

    if dataset == 'mnist':
      self.trainset = torchvision.datasets.MNIST(
                      root=path, train=True,
                      download=download, transform=transform)

      self.testset = torchvision.datasets.MNIST(
                      root=path, train=False,
                      download=download, transform=transform)

    if dataset == 'cifar10':
      self.trainset = torchvision.datasets.CIFAR10(
                      root=path, train=True,
                      download=download, transform=transform)

      self.testset = torchvision.datasets.CIFAR10(
                      root=path, train=False,
                      download=download, transform=transform)

    self.trainloader = torch.utils.data.DataLoader(
                self.trainset, batch_size=batch, 
                shuffle=True, num_workers=3)

    self.testloader = torch.utils.data.DataLoader(
                self.testset, batch_size=batch, 
                shuffle=False, num_workers=3)