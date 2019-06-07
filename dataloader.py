import os
from os.path import isdir, exists, abspath, join

import random

import numpy as np
from PIL import Image

from torchvision import transforms
import matplotlib.pyplot as plt


class DataLoader():
    def __init__(self, root_dir='data', batch_size=16, batch_num=100):
        self.batch_size = batch_size

        self.batch_num = batch_num

        self.root_dir = abspath(root_dir)
        self.data_dir = join(self.root_dir, 'train.png')
        self.labels_dir = join(self.root_dir, 'test.png')


    def applyDataAugmentation(self, img, bach_s, cw):

            self.img_l = [transforms.functional.resized_crop(img, random.randint(0,471), random.randint(0,cw), 400, 400, 400) for i in range(bach_s)]

            for i in range(len(self.img_l)):

                # H FLIP
                if random.random() > 0.5:
                    self.img_l[i] = transforms.functional.hflip(self.img_l[i])

                # V FLIP
                if random.random() > 0.5:
                    self.img_l[i] = transforms.functional.vflip(self.img_l[i])

                # Color Jitter
                if random.random() > 0.5:
                    self.img_l[i] = transforms.functional.adjust_hue(self.img_l[i], random.uniform(-0.1,0.1))

                # Rotation
                if random.random() > 0.5:
                    self.img_l[i] = transforms.functional.rotate(self.img_l[i], random.randint(-30, 30))
                    self.img_l[i] = np.asarray(transforms.functional.five_crop(self.img_l[i], 200)[4])
                    self.img_l[i] = Image.fromarray(self.img_l[i])

                # Resize
                if random.random() > 0.5:
                    self.img_l[i] = transforms.functional.resize(self.img_l[i], random.randint(300,500)) 

        
            return self.img_l


    def mask_gen (self, size = 128):

        mask = np.ones((size,size))*255

        for i in range(5):
            width = random.randint(0,119)
            height = random.randint(0,63)

            if random.random() > 0.5:
                mask[height:height+64,width:width+8] = 0

            else:
                mask[width:width+8,height:height+64] = 0

        return mask
            



    def __iter__(self):
       
        if self.mode == 'train':
            src = self.data_dir
            self.batch_num = 100
            crop_width = 450

        elif self.mode == 'test':
            src = self.labels_dir
            self.batch_num = 1
            crop_width = 30


        for i  in range (self.batch_num):
            data_image = Image.open(src)
            
            img_list = self.applyDataAugmentation(data_image, self.batch_size, crop_width)

            img_list = [i.resize((128,128)) for i in img_list]

            train_gt = np.asarray(img_list[0])[np.newaxis, :]

            mask = self.mask_gen()[np.newaxis,:,:, np.newaxis]

            for img in img_list[1:]:
                train_gt = np.concatenate((train_gt, np.asarray(img)[np.newaxis,:]), axis=0)

                mask2 = self.mask_gen()[np.newaxis,:,:, np.newaxis]
                mask = np.concatenate((mask, mask2 ), axis = 0)

            
            train_in = np.concatenate((train_gt, mask), axis = 3)
            train_in = (train_in * (mask / 255 ))


            train_in = train_in / 255
            train_gt = train_gt / 255

            yield (train_in, train_gt)

    def setMode(self, mode):
        self.mode = mode
