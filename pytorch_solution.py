import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch import nn,optim
from torch.autograd import Variable
from tqdm import tqdm
import time

BATCH_SIZE = 128
NUM_EPOCHS = 10
# preprocessing
normalize = transforms.Normalize(mean=[.5], std=[.5])
transform = transforms.Compose([transforms.ToTensor(), normalize])

# download and load the data
train_dataset = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=transform, download=False)

# encapsulate them into dataloader form
train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

class SimpleNet(nn.Module):
    # TODO:define model
    def __init__(self):
        super(SimpleNet,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(1,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(stride=2,kernel_size=2))
        self.dense=nn.Sequential(
            nn.Linear(14*14*128,1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024,10))
    def forward(self,x):
        x=self.conv(x)
        x=x.view(-1,14*14*128)
        x=self.dense(x)
        return x

model = SimpleNet()
if torch.cuda.is_available():
    model=model.cuda()
# TODO:define loss function and optimiter
criterion = nn.CrossEntropyLoss()
optimizer =torch.optim.Adam(model.parameters())
correct=0
total=0

# train and evaluate


for epoch in range(NUM_EPOCHS):
    for images, labels in tqdm(train_loader):
        # TODO:forward + backward + optimize
        if torch.cuda.is_available():
            images=Variable(images).cuda()
            labels=Variable(labels).cuda()
        else:
            images=Variable(images)
            labels=Variable(labels)
    
        out=model(images)
        loss=criterion(out,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        predicted=torch.argmax(out,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
    
correct=0
total=0
for images, labels in tqdm(test_loader):
    if torch.cuda.is_available():
        images=Variable(images).cuda()
        labels=Variable(labels).cuda()
    preds=model(images)
    predicted=torch.argmax(preds,1)
    total+=labels.size(0)
    correct+=(predicted==labels).sum().item()
test_accuracy=correct/total
print('test_accuracy:{}'.format(test_accuracy))
    # evaluate
    # TODO:calculate the accuracy using traning and testing dataset
