import os
import argparse
import importlib

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torchvision import transforms

from models.EggClassifierV1 import EggClassifierV1
from models.EggClassifierV2 import EggClassifierV2
from datasets.EggDataset import EggDataset
from torch.utils.data import DataLoader

def train(model, optimizer, train_loader, val_loader, scheduler, num_epochs=10,  device='cuda'):
    model = model.to(device)
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        model.train()  # switch to training mode
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_acc += torch.sum(preds == labels.data)

        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)

        print('Epoch [{}/{}] train loss: {:.4f} train acc: {:.4f}'.format(epoch + 1, num_epochs, train_loss, train_acc))
        # validate model
        v_loss, v_acc = validation(model, val_loader, scheduler, device)
    return model, optimizer, train_loss, train_acc, v_loss, v_acc

def validation(model, val_loader, scheduler, device='cuda'):
    val_loss = 0.0
    val_acc = 0.0
    model = model.to(device)
    model.eval()  # switch to evaluation mode
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_acc += torch.sum(preds == labels.data)
    
    scheduler.step(val_loss)

    val_loss /= len(val_loader.dataset)
    val_acc /= len(val_loader.dataset)
    print('Validation loss: {:.4f} Validation acc: {:.4f}'.format(val_loss, val_acc))
    return val_loss, val_acc

def save_model(model, optimizer, save_dir, epoch):
    model_path = os.path.join(save_dir, 'checkpoints', f'epoch_{epoch}.pth')
    complete_model_path = os.path.join(save_dir, 'checkpoints', f'complete_model.pth')
    # create directory if it doesn't exist
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, model_path)
    torch.save(model, model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/media/hdd/mahmood/datasets/EggCompetetion')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--image_channels', type=int, default=3)
    parser.add_argument('--model', type=str, default='EggClassifierV1')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='work_dirs/egg_classifier_v1')
    args = parser.parse_args()

    # define transforms
    train_transforms = transforms.Compose([
        transforms.Resize((args.img_size + 64, args.img_size + 64)),
        transforms.CenterCrop((args.img_size, args.img_size)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),

        # color transforms
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        
        transforms.ToTensor()
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])

    # define datasets
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'test')
    train_dataset = EggDataset(train_dir, transforms=train_transforms)
    val_dataset = EggDataset(val_dir, transforms=val_transforms)

    # define dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # define model and import model class based on name
    model_class = getattr(importlib.import_module('models'), args.model)
    model = model_class(args.num_classes, args.image_channels)

    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # define scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # move model to device
    device = torch.device(args.device)

    # train model
    t_model, t_optimizer, t_loss, t_acc, v_loss, v_acc = train(model, optimizer, train_loader, val_loader, scheduler, args.num_epochs, device)

    # save model
    save_model(t_model, t_optimizer, args.save_dir, args.num_epochs)

    