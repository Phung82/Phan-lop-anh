# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 00:50:28 2020

@author: SB
"""

import os
import torch
import torchvision
import tarfile
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

import torch.nn as nn
import torch.nn.functional as F


import matplotlib.pyplot as plt

from torch.utils.data.dataloader import DataLoader

from torchvision.utils import make_grid


"""-----------------------------------------------------------------"""
def ketnoi():
    data_dir = './data/cifar10'
    print(os.listdir(data_dir))
    classes = os.listdir(data_dir + "/train")
    print(classes)
    dataset = ImageFolder(data_dir+'/train', transform=ToTensor())
    
    img, label = dataset[0]
    print(img.shape, label)
    img
    print('dataset.classes: ',dataset.classes)
"""-----------------------------------------------------------------"""
'''
def show_example(img, label):
    print('Label: ', dataset.classes[label], "("+str(label)+")")
    plt.imshow(img.permute(1, 2, 0))
'''
"""-----------------------------------------------------------------"""
def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        break


"""-----------------------------------------------------------------"""
def apply_kernel(image, kernel):
    ri, ci = image.shape       # image dimensions
    rk, ck = kernel.shape      # kernel dimensions
    ro, co = ri-rk+1, ci-ck+1  # output dimensions
    output = torch.zeros([ro, co])
    for i in range(ro): 
        for j in range(co):
            output[i,j] = torch.sum(image[i:i+rk,j:j+ck] * kernel)
    return output
"""-----------------------------------------------------------------"""
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
"""-----------------------------------------------------------------"""
class Cifar10CnnModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

            nn.Flatten(), 
            nn.Linear(256*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10))
        
    def forward(self, xb):
        return self.network(xb)
    
"""-------------------DDAFO TAJO-------------------------"""
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    print("Loading......!")
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history

"""-----------------------------------------------------------------"""

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
# đưa mô hình vào thiết bị GPU được chọn
def to_device(data, device):

    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
"""-----------------------------------------------------------------"""
def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');

"""---------------test mohinh--------------------------------"""

def predict_image(img, model):
    # Convert to a batch of 1
    device = get_default_device()
    print(device)
    data_dir = './data/cifar10'
    print(os.listdir(data_dir))
    classes = os.listdir(data_dir + "/train")
    print(classes)
    dataset = ImageFolder(data_dir+'/train', transform=ToTensor())
    xb = to_device(img.unsqueeze(0),device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    print(classes[preds[0].item()])
    return dataset.classes[preds[0].item()]
"""-----------------------------------------------------------------"""

'''https://www.programcreek.com/python/example/105102/torchvision.datasets.ImageFolder'''
"""-----------------------------------------------------------------"""
def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss/accuracy')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs');
"""-----------------------------------------------------------------"""


"""-----------------------------------------------------------------"""


"""-----------------------------------------------------------------"""


"""-----------------------------------------------------------------"""

"""-----------------------------------------------------------------"""


"""-----------------------------------------------------------------"""






def main():
    device = get_default_device()
    print(device)
    #ketnoi()
    data_dir ='C:/Users/SB/Desktop/Do-An-MNR/data/cifar10'
    print(os.listdir(data_dir))
    classes = os.listdir(data_dir + "/train")
    print(classes)
    dataset = ImageFolder(data_dir+'/train', transform=ToTensor())
    img, label = dataset[0]
    print(img.shape, label)
    img
    print('dataset.classes: ',dataset.classes)
    #show_example(*dataset[0])
    #dao taao va xac thuc
    random_seed = 42 
    torch.manual_seed(random_seed);
    val_size = 5000
    train_size = len(dataset) - val_size
    
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print("len(train_ds), len(val_ds): ",len(train_ds), len(val_ds),)
    print("Run 0!")
    batch_size=128    
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)
    
    #show_batch(train_dl)
    
    #train model doane
    model = Cifar10CnnModel()
    model
    
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    to_device(model, device);
    print("Run 1!")
    model = to_device(Cifar10CnnModel(), device)
    evaluate(model, val_dl)
    print("Run 2!")
    
    num_epochs = 20
    opt_func = torch.optim.Adam
    lr = 0.001
    
    history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
    
    print("Run 3!")
    plot_accuracies(history)
    plot_losses(history)
    print("Run 4!")
    #luu mo hinh
    PATH = './cifar_e20.pth'
    torch.save(model.state_dict(), PATH)
    '''
    s='/test/cat/0972.png'

    S=s.split('/')
    print(S)
    So=S[len(S)-2]
    print(So)
    P=S[len(S)-1].split('.png')
    print(P[0])
    
    if So=='frog':
    	N=int(2000)
    if So=='cat':
    	N=int(1000)
    if So=='bird':
    	N=int(0)
    if So=='horse':
    	N=int(3000)
    
    kq=int(P[0])+N-1
    print("Run 5!")
    model2 = to_device(Cifar10CnnModel(), device)
    model2.load_state_dict(torch.load('cifar_net.pth'))
    test_dataset = ImageFolder(data_dir+'/test', transform=ToTensor())
    img, label = test_dataset[kq]
    #print('IMG: ',type(label))
    #label =2
    #img="test/frog/0021.png"
    plt.imshow(img.permute(1, 2, 0))
    print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model2))
    print('Nhan: ',type(predict_image(img, model2)))
    '''
    print("Run main!")
    
    
    
if __name__ == '__main__':
    main()
    
print("Done!")
    
    
    
    
    
    
    
    
    
