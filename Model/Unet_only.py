'''
Author: AKiy github:Akiy071
Date: 2024-02-29 15:07:16
LastEditors: AKiy
LastEditTime: 2024-05-01 11:59:11
Description: Unet-only model.
'''
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torchsummary import summary
import cv2
import matplotlib.pyplot as plt
import sys
sys.path.append(r"Z:\Desktop\python项目\03_BEiTwithAttention")
sys.path.append(r"Z:\Desktop\python项目\03_BEiTwithAttention\DataProcess")

from DataProcess.DataSet import MyDataSet

def DoubleConv(in_channels, out_channels, mid_channels=None):
    if not mid_channels:
        mid_channels = out_channels
    sequential = nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, 3, padding=1),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    return sequential

def DownScaling(in_channels, out_channels):
    sequential = nn.Sequential(
        nn.MaxPool2d(2),
        DoubleConv(in_channels, out_channels)
    )
    return sequential

class UpScaling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpScaling, self).__init__()

        self.sequential = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        """
        self.sequential = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        """
        self.Conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x, concat_x):
        x = self.sequential(x)
        # 插值
        Y = concat_x.size()[2]-x.size()[2]
        X = concat_x.size()[3]-x.size()[3]

        x = F.pad(x, [Y//2, X-X//2,
                      Y//2, Y-Y//2])

        x = torch.cat([concat_x, x], dim=1)
        x = self.Conv(x)
        return x
    
class MyNet(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs) -> None:
        super(MyNet, self).__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input=DoubleConv(self.in_channels,32)
        self.layer1=DownScaling(32,64)
        self.layer2=DownScaling(64,128)
        self.layer3 = DownScaling(128, 256)
        self.layer4 = DownScaling(256, 512)

        self.layer5 = UpScaling(512, 256)
        self.layer6 = UpScaling(256, 128)
        self.layer7 = UpScaling(128, 64)
        self.layer8 = UpScaling(64, 32)
        
        self.output = nn.Conv2d(32, out_channels, kernel_size=1)
        
    def forward(self,x):
        
        x1=self.input(x)
        
        x2 = self.layer1(x1)        
        x3 = self.layer2(x2)
        x3 = nn.Dropout(0.4)(x3)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x5 = nn.Dropout(0.4)(x5)

        x = self.layer5(x5, x4)
        x = self.layer6(x4, x3)
        x=nn.Dropout(0.4)(x)
        x = self.layer7(x, x2)
        x=nn.Dropout(0.4)(x)
        
        x = self.layer8(x, x1)
        
        concat_x = self.output(x)
        
        return concat_x
        
def norm_image(out):
    maxvalue = out.max()
    out = out*255/maxvalue
    mat = np.uint8(out)
    mat = mat.transpose(1, 2, 0)
    return mat

def mask_heatImage(image, mask):
    masks = norm_image(mask).astype(np.uint8)
    heatmap = cv2.applyColorMap(masks, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap)
    cam = 0.4*heatmap+0.6*np.float32(image)
    return np.uint8(cam)

def initalize_model(model):
    for name,param in model.named_parameters():
        if "weight" in name and param.data.ndim >=2:
            nn.init.xavier_uniform_(param.data)
            if "relu" in name.lower():
                nn.init.kaiming_uniform_(param.data,a=0,mode="fan_in",nonlinearity="relu")
        elif "bias" in name:
            nn.init.constant_(param.data,0)
            
def collate_fn(bacth,nums=4):
    img_storage=[]
    lb_storage=[]
    img=[]
    lb=[]
    # get patch [p,3,patch_size,patch_size]
    for i in range(len(bacth)):
        for images in bacth[i][0]:
            images=torch.from_numpy(images)
            if len(img)<nums:
                img.append(images)
            else:
                img_storage.append(images)
        for labels in bacth[i][1]:
            labels=torch.from_numpy(labels) #numpy to tensor
            labels=torch.where(labels[2]>0,1,0)
            if len(lb)<nums:
                lb.append(labels)
            else:
                lb_storage.append(labels)
        
    if len(img)<nums:
        while len(img)<nums and len(img_storage)>0 and len(lb_storage)>0:
            img.append(img_storage.pop(0))
            lb.append(lb_storage.pop(0))
    
    return torch.stack(img),torch.stack(lb)


def Show(image,label,pre):
    mean=[0.485,0.456,0.406]
    std=[0.229,0.224,0.225]
    image=image.cpu().numpy().transpose(1,2,0)
    image=((image*std+mean)*255).astype("uint8")
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    label=label.numpy().astype(np.uint8)
    # label=cv2.cvtColor(label.transpose(1,2,0),cv2.COLOR_BGR2RGB)
    pre=mask_heatImage(image,pre)
    pre=cv2.cvtColor(pre,cv2.COLOR_BGR2RGB)
    lst_name=["image","label","pre"]
    lst=[image,label,pre]
    plt.figure(figsize=(10,10))
    for i in range(3):    
        plt.subplot(1,3,i+1)
        plt.imshow(lst[i])
        plt.title(lst_name[i])
    plt.show()

if __name__=="__main__":
    x=torch.rand((1,3,256,256)).cuda()
    model=MyNet(3,2).cuda() # outchannel is number of your class.
    summary(model,(3,256,256))
    outputs=model(x)
    print(outputs.shape)