import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

def get_mean_std(loader):
  channels_sum, channels_squared_sum, num_batches = 0, 0, 0

  knew = 1
  for data, _ in loader:
    if knew==1:
      print(data.shape)
      knew = 0
    channels_sum += torch.mean(data, dim=[0,2,3])
    channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
    num_batches += 1

  mean = channels_sum/num_batches
  std = (channels_squared_sum/num_batches - mean**2)**0.5

  return mean, std


mean, std = get_mean_std(train_loader)

print(mean)
print(std)