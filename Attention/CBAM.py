'''
Author: AKiy github:Akiy071
Date: 2024-03-26 13:19:37
LastEditors: AKiy
LastEditTime: 2024-03-26 13:32:52
reference from https://blog.csdn.net/m0_45447650/article/details/123983483
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import math
from torchsummary import summary

class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=3, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
 
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
 
        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
 
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
 
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

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

# if __name__ == '__main__':
#     import cv2
#     import numpy as np
#     Test_Img = cv2.imread(
#         r"DataProcess\output\test\01_200906113326_490000_180_crop_2.jpg")
#     x = torch.tensor(Test_Img).unsqueeze(0).permute(0,3,1,2).float().cuda()
#     print(x.shape)
#     model = CBAMLayer(3).cuda()
#     x = nn.Conv2d(3, 3, kernel_size=1).cuda()(x)
#     out = model(x)
#     print(out.shape)

#     # visiualize heatmap about ccAttention
#     out = out.cpu().squeeze(0).detach().numpy()
#     out = mask_heatImage(Test_Img, out)
#     cv2.imshow("result", out)
#     cv2.waitKey(0)