import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torchvision import datasets, transforms



train_dataset = datasets.CIFAR10(root='./data', download=True, train=True,  \
                                  transform=transforms.Compose([transforms.ToTensor(), \
                                  transforms.Normalize((0.4914, 0.4822, 0.4466), (0.2470, 0.2435, 0.2616)) \
                                  ]))
val_dataset = datasets.CIFAR10(root='./data', download=False, train=False,  \
                                  transform=transforms.Compose([transforms.ToTensor(), \
                                  transforms.Normalize((0.4914, 0.4822, 0.4466), (0.2470, 0.2435, 0.2616)) \
                                  ]))

batch_size = 10
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

use_cuda = torch.cuda.is_available()

class CNNClassifier(nn.Module):

  def __init__(self):

    super(CNNClassifier, self).__init__()
    conv1 = nn.Conv2d(3, 6, 5)
    pool = nn.MaxPool2d(2, 2)
    conv2 = nn.Conv2d(6, 16, 5)
    fc1 = nn.Linear(16 * 5 * 5, 120)
    fc2 = nn.Linear(120, 84)
    fc3 = nn.Linear(84, 10)

    self.conv_module = nn.Sequential(
      conv1,
      nn.ReLU(),
      pool,
      conv2,
      nn.ReLU(),
      pool
    )

    self.fc_module = nn.Sequential(
      fc1,
      nn.ReLU(),
      fc2,
      nn.ReLU(),
      fc3
    )

    if use_cuda:
      print('using gpu!')

      self.conv_module = self.conv_module.cuda()
      self.fc_module = self.fc_module.cuda()


  def forward(self, x):
    out = self.conv_module(x)
    out = torch.flatten(out, 1)
    out = self.fc_module(out)
    return F.softmax(out, dim=1)

cnn = CNNClassifier()

criterion = nn.CrossEntropyLoss()
learning_rate = 1e-3
#optimizer = optim.SGD(cnn.parameters(), lr=learning_rate, momentum=0.9)    
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate) 



for epoch in range(50):

  running_loss = 0.0
  for i, data in enumerate(train_loader, 0):

    inputs, labels = data
    
    if use_cuda:
      inputs, labels = inputs.cuda(), labels.cuda()

    optimizer.zero_grad()

    outputs = cnn(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    if i % 2000 == 1999:
      print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
      running_loss = 0.0

print('Finished Training')

PATH = './cifar_net.path'
torch.save(cnn.state_dict(), PATH)

correct = 0
total = 0

with torch.no_grad():
  for data in val_loader:
    images, labels = data
    if use_cuda:
      images, labels = images.cuda(), labels.cuda()
    
    outputs = cnn(images)

    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
      




    
  