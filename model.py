
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.maxpool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.downstep1 = downStep(4, 64)
        self.downstep2 = downStep(64,128)
        self.downstep3 = downStep(128,256)
        self.downstep4 = downStep(256,512)
        self.downstep5 = downStep(512,1024)

        self.upstep1 = upStep(1024,512,True)
        self.upstep2 = upStep(512,256,True)
        self.upstep3 = upStep(256,128,True)
        self.upstep4 = upStep(128,64,False)

    def forward(self, x):
       
        x1 = self.downstep1(x)
        x2 = self.maxpool1(x1)
        x3 = self.downstep2(x2)
        x4 = self.maxpool1(x3)
        x5 = self.downstep3(x4)
        x6 = self.maxpool1(x5)
        x7 = self.downstep4(x6)
        x8 = self.maxpool1(x7)
        x9 = self.downstep5(x8)

        x = self.upstep1(x9, x7)
        x = self.upstep2(x, x5)
        x = self.upstep3(x, x3)
        x = self.upstep4(x, x1)

        return x


class downStep(nn.Module):
    def __init__(self, inC, outC):
        super(downStep, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels = inC, out_channels = outC, kernel_size = 3, padding= 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels = outC, out_channels = outC, kernel_size = 3, padding= 1)

    def forward(self, x):
       
        x = self.conv1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        return x


class upStep(nn.Module):
    def __init__(self, inC, outC, withReLU=True):
        super(upStep, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(in_channels = inC, out_channels = outC , kernel_size = 2, stride = 2)
        self.conv1 = nn.Conv2d(in_channels = inC, out_channels = outC, kernel_size = 3, padding= 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels = outC, out_channels = outC, kernel_size = 3, padding= 1)
        self.conv3 = nn.Conv2d(in_channels = outC , out_channels = 3, kernel_size = 1)
        self.withReLU = withReLU
        self.sigmoid = nn.Sigmoid()

    def center_crop(self, layer, target_size):
        _, _, layer_heigh, layer_width = layer.size()
        diff_y = (layer_heigh - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, x_down):
        
        x = self.deconv1(x)
        x = torch.cat([x, x_down], 1)
        x = self.conv1(x)
        if self.withReLU:
            x = self.relu(x)
            
        x = self.conv2(x)
        if self.withReLU:
            x = self.relu(x)
            
        else :
            x = self.conv3(x)
            x = self.sigmoid(x)
        return x