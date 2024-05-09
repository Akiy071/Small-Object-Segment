'''
Author: AKiy github:Akiy071
Date: 2024-03-26 14:46:12
LastEditors: AKiy
LastEditTime: 2024-04-24 11:04:09
Description: we decide to bulid the yolov5n which is smallest consumption version.
'''
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torchsummary import summary
import cv2
import matplotlib.pyplot as plt
import sys
sys.path.append(r".\03_BEiTwithAttention")

from Attention.CBAM2 import SpatialAttention,ChannelAttention 

class Conv(nn.Module):
    def __init__(self,inchannel,outchannel,k=1,s=1,p=0) -> None:
        super().__init__()
        self.Conv=nn.Conv2d(inchannel,outchannel,kernel_size=k,stride=s,padding=p)
        self.BN=nn.BatchNorm2d(outchannel)
        self.SiLU=nn.SiLU(inplace=True) 
    
    def forward(self,x):
        x=self.Conv(x)
        x=self.BN(x)
        x=self.SiLU(x)
        
        return x
    
class BottleNeck(nn.Module):
    def __init__(self,inchannel,outchannel,short_cut=True,pad=1,e=0.5) -> None:
        super(BottleNeck,self).__init__()
        hidden_layer=int(outchannel*e) # Set hidden layers num
        self.conv1=Conv(inchannel,hidden_layer,1,1)
        self.conv2=Conv(hidden_layer,outchannel,3,1,pad)
        self.add=short_cut and inchannel==outchannel
        
    def forward(self,x):
        x1=self.conv1(x)
        x2=self.conv2(x1)

        return x+x2 if self.add else x2

class C3(nn.Module):
    def __init__(self,inchannel,outchannel,n=1,short_cut=True,pad=1,e=0.5) -> None:
        super(C3,self).__init__()
        hidden_layer=int(outchannel*e) # Set hidden layers num
        self.Conv1=Conv(inchannel,hidden_layer,1,1)
        self.Conv2=Conv(inchannel,hidden_layer,1,1)
        self.Conv3=Conv(2*hidden_layer,outchannel,1)
        self.m=nn.Sequential(*[BottleNeck(hidden_layer,hidden_layer,short_cut,pad,e=1.0) for _ in range(n)])
    
    def forward(self,x):
        x1=self.Conv1(x)
        x1=self.m(x1)
        x2=self.Conv2(x)
        x3=torch.cat((x1,x2),dim=1)
        x4=self.Conv3(x3)
        return x4

class SPPF(nn.Module):
    def __init__(self,inchannel,outchannel,k=5) -> None:
        super(SPPF,self).__init__()
        hidden_layer=int(outchannel//2)
        self.Conv1=Conv(inchannel,hidden_layer,1,1)
        self.Conv2=Conv(hidden_layer*4,outchannel,1,1)
        self.Maxpool=nn.MaxPool2d(kernel_size=k,stride=1,padding=k//2)
        
    def forward(self,x):
        
        x=self.Conv1(x)
        
        y1=self.Maxpool(x)
        y2=self.Maxpool(y1)
        
        x1=torch.cat([x,y1,y2,self.Maxpool(y2)],dim=1)
        x2=self.Conv2(x1)
        return x2


class BackBone(nn.Module):
    def __init__(self, inchannel,hidden_layer) -> None:
        super(BackBone,self).__init__()
        
        # if hidden_layer=64
        self.Conv1=Conv(inchannel,hidden_layer,6,2,2)   # 3->64 
        self.Conv2=Conv(hidden_layer,hidden_layer*2,k=3,s=2,p=1) # 64->128
        self.C3_1=C3(hidden_layer*2,hidden_layer*2,n=1,short_cut=True) # 128->128
        self.Conv3=Conv(hidden_layer*2,hidden_layer*4,k=3,s=2,p=1) # 128->256
        
        self.Attention1=ChannelAttention(128)
        self.Attention2=SpatialAttention()
        self.Attention3=ChannelAttention(256)
        self.Attention4=SpatialAttention()
        self.Attention5=ChannelAttention(512)
        self.Attention6=SpatialAttention()
        
        # Short-cat
        self.C3_2=C3(hidden_layer*4,hidden_layer*4,n=2,short_cut=True) # 256->256
        self.Conv4=Conv(hidden_layer*4,hidden_layer*8,k=3,s=2,p=1) # 256->512
        
        #Short-cat
        self.C3_3=C3(hidden_layer*8,hidden_layer*8,n=3,short_cut=True) # 512->512
        self.Conv5=Conv(hidden_layer*8,hidden_layer*16,k=3,s=2,p=1) # 512->1024
        self.C3_4=C3(hidden_layer*16,hidden_layer*16,n=1,short_cut=True) # 1024->1024
        
        #Short-cat
        self.SPPF=SPPF(hidden_layer*16,hidden_layer*16,k=5) # 1024->1024
    
    def forward(self,x):
        x=self.Conv1(x)
        x=self.Conv2(x)
        x=self.C3_1(x)
        x=self.Conv3(x)
        
        #Add Attention block before get feature map 
        x=self.Attention1(x)*x
        x=self.Attention2(x)*x
        
        concat_x1=self.C3_2(x)
        x1=self.Conv4(concat_x1)
        
        x1=self.Attention3(x1)*x1
        x1=self.Attention4(x1)*x1
        
        concat_x2=self.C3_3(x1)
        x2=self.Conv5(concat_x2)
        x2=self.C3_4(x2)
        
        x2=self.Attention5(x2)*x2
        x2=self.Attention6(x2)*x2 
        
        output=self.SPPF(x2)

        return [output,concat_x2,concat_x1]

class NeckWithHead(nn.Module):
    def __init__(self, inchannel,hidden_layer) -> None:
        super(NeckWithHead,self).__init__()
        self.Backbone=BackBone(inchannel,hidden_layer) # [1024,512,256]
        
        # connect self.Backbone outpt
        self.Conv1=Conv(hidden_layer*16,hidden_layer*8,k=1,s=1) # 1024->512
        self.UpSample=nn.Upsample(None,2,"nearest") # 512->512
        
        # connect self.Backbone concat_x2
        self.C3_1=C3(hidden_layer*16,hidden_layer*8,n=1,short_cut=False) # after concat 1024->512
        self.Conv2=Conv(hidden_layer*8,hidden_layer*4,k=1,s=1) # 512->256
        
        # connect self.Backbone Concat_x1
        self.C3_2=C3(hidden_layer*8,hidden_layer*4,n=1,short_cut=False) # after upsample and concat: 512->256
        self.Conv3=Conv(hidden_layer*4,hidden_layer*4,k=3,s=2,p=1) # 256->256
        
        # concat with self.conv2
        self.C3_3=C3(hidden_layer*8,hidden_layer*8,n=1,short_cut=False) # after concat: 512->512
        self.Conv4=Conv(hidden_layer*8,hidden_layer*8,k=3,s=2,p=1) # 512->512
        
        # concat with self.conv1
        self.C3_4=C3(hidden_layer*16,hidden_layer*16,n=1,short_cut=False) # after concat:1024->1024
        
        self.output_1=Conv(hidden_layer*4,hidden_layer*4,k=1,s=1) # Small 256
        self.output_2=Conv(hidden_layer*8,hidden_layer*8,k=1,s=1) # Medium 512
        self.output_3=Conv(hidden_layer*16,hidden_layer*16,k=1,s=1) # large 1024
        
    def forward(self,x):
        
        concat_x3,concat_x2,concat_x1=self.Backbone(x)
        concat_x3=self.Conv1(concat_x3)
        
        x3=self.UpSample(concat_x3)
        x3=torch.concat((x3,concat_x2),dim=1)
        x3=self.C3_1(x3)
        concat_x2=self.Conv2(x3)
        
        x4=self.UpSample(concat_x2)
        concat_x1=torch.concat((x4,concat_x1),dim=1)
        
        small=self.C3_2(concat_x1)
        
        x5=self.Conv3(small)
        x5=torch.concat((x5,concat_x2),dim=1)
        
        medium=self.C3_3(x5)
        
        x6=self.Conv4(medium)
        concat_x3=torch.concat((x6,concat_x3),dim=1)
        
        large=self.C3_4(concat_x3)
        
        # output
        large=self.output_3(large)
        medium=self.output_2(medium)
        small=self.output_1(small)
    
        return [small,medium,large] # Get three feature map.

class UpScaling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpScaling, self).__init__()

        self.Conv=Conv(in_channel,out_channel,k=1,s=1)
        self.C3 = C3(in_channel,out_channel,n=1,short_cut=False)
        
    def forward(self, x, concat_x):
        x = self.Conv(x)
        # 插值
        Y = concat_x.size()[2]-x.size()[2]
        X = concat_x.size()[3]-x.size()[3]

        x = F.pad(x, [Y//2, X-X//2,
                      Y//2, Y-Y//2])

        x = torch.cat([concat_x, x], dim=1)
        x = self.C3(x)
        return x

class Segment(nn.Module):
    def __init__(self,inchannel,hidden_layer,outchannel) -> None:
        super(Segment,self).__init__()
        
        self.neckwithhead=NeckWithHead(inchannel,hidden_layer)
        
        # This part may not perfect,but it decide on you.
        self.UpScaling1=UpScaling(hidden_layer*16,hidden_layer*8)
        self.UpScaling2=UpScaling(hidden_layer*8,hidden_layer*4)
        self.Upsample=nn.Upsample(None,2,"nearest")
        self.Conv_1=Conv(hidden_layer*4,hidden_layer*2,3,1,1)
        self.C3_1=C3(hidden_layer*2,hidden_layer*2,n=1,short_cut=False)
        self.Conv_2=Conv(hidden_layer*2,hidden_layer,3,1,1)
        self.C3_2=C3(hidden_layer,hidden_layer,n=1,short_cut=True)
        self.Conv=nn.Conv2d(hidden_layer,outchannel,1,1)
    
    def forward(self,x):
        
        x=self.neckwithhead(x)
        x1=x[2] # Using  large feature map which is more detail of data.   
        x2=self.UpScaling1(x1,x[1])
        x3=self.UpScaling2(x2,x[0])
        
        x3=self.Conv_1(x3)
        x3=self.Upsample(x3)
        x4=self.C3_1(x3)
        
        x4=self.Conv_2(x4)
        x4=self.Upsample(x4)
        x4=self.C3_2(x4)
        
        x5=self.Upsample(x4)
        output=self.Conv(x5)
        
        return output
        

if __name__=="__main__":
    x=torch.rand((1,3,256,256)).cuda()
    model=Segment(3,32,2).cuda() # outchannel is number of your class.
    summary(model,(3,256,256))
    outputs=model(x)
    print(outputs.shape)
