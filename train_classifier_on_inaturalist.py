# We retrain the classifier used in the style transfer using
# https://github.com/visipedia/inat_comp/tree/master/2021
# to allow us to do style transfer with a network that has never seen human artworks.
# Specifically we trained the network to classify the "order" (273 categories)
# in the "train mini" subset (42GB) of the iNaturalist dataset


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import time
import shutil
import os
import numpy as np
import sys
import argparse

import config


#target_type='kingdom'
#target_type='phylum'
target_type='order'
#target_type='full'


depth=19
_num_classes_={'kingdom':3,'phylum':13,'full':10000,'order':300}
num_classes=_num_classes_[target_type]

outdir=f'{config.outdir}/{depth}_trained_on_{target_type}'
os.makedirs(outdir,exist_ok=True)


device = torch.device('cuda')
num_epochs = 100
learning_rate = .001
input_size=224

# Data augmentation and normalization for training
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((input_size,input_size)),  # scale imported image
        #transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

def getdataset(download):
    return torchvision.datasets.INaturalist(root=config.datapath,
                                            #version='2021_train',
                                            version='2021_train_mini',
                                            target_type=target_type,
                                            transform=data_transforms['train'],
                                            download=download,
                                            )

try:
    # load previously downloaded dataset
    train_dataset=getdataset(False)
except Exception as e:
    #first run
    train_dataset=getdataset(True)



torch.manual_seed(0)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=50, shuffle=True)
criterion = nn.CrossEntropyLoss()
total_step = len(train_loader)

os.makedirs(outdir,exist_ok=True)
# first run
if depth==11:
    model = models.vgg11(weights=None, num_classes=num_classes).to(device)
if depth==16:
    model = models.vgg16(weights=None, num_classes=num_classes).to(device)
if depth==19:
    model = models.vgg19(weights=None, num_classes=num_classes).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
start_epoch=0

def unpack(a):
    try: return a.item()
    except: return a


for epoch in range(start_epoch, num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        stats=f"epoch {epoch+1} i {i+1} loss {loss.item()} learning_rate {learning_rate}"

        with open(os.path.join(outdir,'loss.txt'),'a') as f:
            f.write(stats+"\n")
        
        if (i+1) % 10 == 0:
            print(stats)

        if (i+1)%100==0:
            checkpoint={
                    'model': model,
                    'epoch': epoch,
                    'i': i+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }
            torch.save(
                checkpoint, 
                os.path.join(outdir,f'checkpoint_{epoch}_{i+1}.pth')
                )
            torch.save(
                model,
                os.path.join(outdir,f'model_{epoch}_{i+1}.pth')
                )
            
    checkpoint={
            'model': model,
            'epoch': epoch+1,
            'i': 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }
    torch.save(
        checkpoint, 
        os.path.join(outdir,f'checkpoint_epoch_{epoch+1}.pth')
        )
    torch.save(
        model,
        os.path.join(outdir,f'model_epoch_{epoch+1}.pth')
        )
