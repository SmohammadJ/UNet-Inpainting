import sys
import os
from os.path import join
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

from torchvision import transforms

import matplotlib.pyplot as plt

from model import UNet
from dataloader import DataLoader

def train_net(net,
              epochs=100,
              data_dir='data',
              lr=0.00001,
              save_cp=True,
              gpu=False):

    if gpu:
        net.cuda()


    print("epochs : ", epochs )
    
    loader = DataLoader(data_dir)

    optimizer = optim.Adam(net.parameters(),
                            lr=lr,
                            weight_decay=0.05)

    for epoch in range(epochs):
        
        print('Epoch %d/%d' % (epoch + 1, epochs))
        print('Training...')
        
        net.train()
        loader.setMode('train')
    
        epoch_loss = 0
    
        for i, (img, label) in enumerate(loader):

            # Tensors
            img_tensor = torch.from_numpy(img).float()
            img_tensor = img_tensor.permute(0,3,1,2) 

            label_tensor = torch.from_numpy(label).float()
            label_tensor = label_tensor.permute(0,3,1,2)
            
    
            # load image/label tensor to gpu
            if gpu:
                img_tensor = img_tensor.cuda()
                label_tensor = label_tensor.cuda()
    
            # get prediction and getLoss()
            optimizer.zero_grad()
            pred = net.forward(img_tensor) 

            loss = getLoss(pred, label_tensor)

            epoch_loss += loss.item()
    
            print('Training sample %d / 100  - Loss: %.6f' % (i+1, loss.item()))
    
            # optimize weights
            loss.backward()
            optimizer.step()

        # torch.save(net.state_dict(), join(data_dir, 'checkpoints') + '\CP%d.pth' % (epoch + 1))
        print('Checkpoint %d saved !' % (epoch + 1))
        print('Epoch %d finished! - Loss: %.6f' % (epoch+1, epoch_loss / (i+1)))

        train_gt = f'{epoch+1}_train_gt.png'
        label_show = label_tensor.cpu().detach().permute(0,2,3,1)
        plt.imsave(join(data_dir, 'samples', train_gt), label_show[15])

        train_in = f'{epoch+1}_train_in.png'
        img_show = img_tensor.cpu().detach().permute(0,2,3,1)
        plt.imsave(join(data_dir, 'samples', train_in), img_show[15,:,:,:3])

        train_out = f'{epoch+1}_train_out.png'
        pred_show = pred.cpu().detach().permute(0,2,3,1)
        plt.imsave(join(data_dir, 'samples', train_out), pred_show[15])



        

        loader.setMode('test')
        net.eval()
        with torch.no_grad():
            for i, (img,label) in enumerate(loader):

                img_tensor = torch.from_numpy(img).float()
                img_tensor = img_tensor.permute(0,3,1,2) 

                label_tensor = torch.from_numpy(label).float()
                label_tensor = label_tensor.permute(0,3,1,2)

                if gpu:
                    img_tensor = img_tensor.cuda()
                    label_tensor = label_tensor.cuda()

                pred = net.forward(img_tensor)

        test_gt = f'{epoch+1}_test_gt.png'
        label_show = label_tensor.cpu().detach().permute(0,2,3,1)
        plt.imsave(join(data_dir, 'samples', test_gt), label_show[15])

        test_in = f'{epoch+1}_test_in.png'
        img_show = img_tensor.cpu().detach().permute(0,2,3,1)
        plt.imsave(join(data_dir, 'samples', test_in), img_show[15,:,:,:3])

        test_out = f'{epoch+1}_test_out.png'
        pred_show_ = pred.cpu().detach().permute(0,2,3,1)
        plt.imsave(join(data_dir, 'samples', test_out), pred_show_[15])

def getLoss(pred_label, target_label):
    loss = nn.MSELoss()
    return loss(pred_label, target_label)

    
def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=100, type='int', help='number of epochs')
    parser.add_option('-d', '--data-dir', dest='data_dir', default='data', help='data directory')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu', default=False, help='use cuda')
    parser.add_option('-l', '--load', dest='load', default=False, help='load file model')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':

    args = get_args()

    net = UNet()

    if args.load:
        if args.gpu:
            net.load_state_dict(torch.load(args.load))
        else:
            net.load_state_dict(torch.load(args.load, map_location='cpu'))
        print('Model loaded from %s' % (args.load))

    if args.gpu:
        net.cuda()
        cudnn.benchmark = True
    
    train_net(net=net,
        epochs=args.epochs,
        gpu=args.gpu,
        data_dir=args.data_dir)
