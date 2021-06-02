import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torchvision import datasets, transforms
import time
import os 
import copy 





train_dataset = datasets.ImageFolder(root='./train',  \
                                  transform=transforms.Compose([ \
                                  transforms.RandomHorizontalFlip(), \
                                  transforms.RandomGrayscale(), \
                                  transforms.ToTensor(), \
                                  transforms.Normalize((0.5556, 0.5069, 0.4563), (0.2828, 0.2740, 0.2886)) \
                                  ]))
val_dataset = datasets.ImageFolder(root='./test',  \
                                  transform=transforms.Compose([transforms.ToTensor(), \
                                  transforms.Normalize((0.5499, 0.5131, 0.4630), (0.2806, 0.2710, 0.2819)) \
                                  ]))

train_size = int(0.8 * len(train_dataset))
test_size = len(train_dataset) - train_size
print(train_size)  

train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])



batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

dataloaders = {'train': train_loader, 'val':val_loader}
dataset_sizes = {'train': train_size ,'val':test_size}

use_cuda = torch.cuda.is_available()

class CNNClassifier(nn.Module):

  def __init__(self):

    super(CNNClassifier, self).__init__()
    conv1 = nn.Conv2d(3, 6, 6)
    pool = nn.MaxPool2d(2, 2)
    bn1 = nn.BatchNorm2d(6)
    conv2 = nn.Conv2d(6, 13, 6)
    fc1 = nn.Linear(13*53*53, 256)
    bn2 = nn.BatchNorm1d(256)
    fc2 = nn.Linear(256, 128)
    bn3 = nn.BatchNorm1d(128)
    fc3 = nn.Linear(128, 7)

    nn.init.kaiming_uniform_(fc1.weight)
    nn.init.kaiming_uniform_(fc2.weight)
    nn.init.kaiming_uniform_(fc3.weight)

    self.dropout = nn.Dropout(0.5)

    self.conv_module = nn.Sequential(
      conv1,
      bn1,
      nn.ReLU(),
      pool,
      conv2,
      nn.ReLU(),
      pool
    )

    self.fc_module = nn.Sequential(
      fc1,
      bn2,
      nn.ReLU(),
      fc2,
      bn3,
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
    #print(out.shape)
    out = self.fc_module(out)
    out = self.dropout(out)
    return F.softmax(out, dim=1)

cnn = CNNClassifier()

criterion = nn.CrossEntropyLoss()
learning_rate = 1e-4
#optimizer = optim.SGD(cnn.parameters(), lr=learning_rate, momentum=0.9)    
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate) 
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.7)



'''for epoch in range(300):

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
    if i % 10 == 9:
      print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 10))
      running_loss = 0.0
'''

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
  since = time.time()

  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0

  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs -1))
    print('-' * 10)

    for phase in ['train', 'val']:
      if phase == 'train':
        model.train()
      else:
        model.eval()

      running_loss = 0.0
      running_corrects = 0

      for inputs, labels in dataloaders[phase]:
        if use_cuda:
          inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
          outputs = model(inputs)
          _, preds = torch.max(outputs, 1)
          loss = criterion(outputs, labels)


          if phase == 'train':
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
      if phase == 'train':
        scheduler.step()
      
      epoch_loss = running_loss / dataset_sizes[phase]
      epoch_acc = running_corrects.double() / dataset_sizes[phase]

      print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

      if phase == 'val' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())


    print()

  print('Best val Acc: {:4f}'.format(best_acc))

  model.load_state_dict(best_model_wts)

  return model

cnn = train_model(cnn, criterion, optimizer, scheduler, num_epochs=50)

print('Finished Training')

PATH = './cifar_net.path'
torch.save(cnn.state_dict(), PATH)

correct = 0
total = 0

with torch.no_grad():
  for data in val_loader:
    images, labels = data
    #print(labels)
    if use_cuda:
      images, labels = images.cuda(), labels.cuda()
    
    outputs = cnn(images)

    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
      




    
  