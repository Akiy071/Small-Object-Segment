'''
Author: AKiy github:Akiy071
Date: 2024-04-17 10:09:05
LastEditors: AKiy
LastEditTime: 2024-04-17 11:37:18
reference from https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import math
from torchsummary import summary
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


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

if __name__ == '__main__':
    import cv2
    import numpy as np
    Test_Img = cv2.imread(
        r"DataProcess\output\test\01_200906113326_490000_180_crop_2.jpg")
    x = torch.tensor(Test_Img).unsqueeze(0).permute(0,3,1,2).float().cpu()
    print(x.shape)
    
    model=nn.Sequential(
        nn.Conv2d(3,64,kernel_size=3,padding=1).cpu(),
        nn.Conv2d(64,3,kernel_size=3,padding=1).cpu()
    )
    
    target_layer = [model]
    
    cam = GradCAM(model=model, target_layers=target_layer)
    targets = None
    grayscale_cam = cam(input_tensor=x, targets=targets)
    grayscale_cam = grayscale_cam[0,:]
    cam_image = show_cam_on_image(x, grayscale_cam, use_rgb=False)
    cv2.imwrite(f'./a.jpeg', cam_image)

    
    out=model(x)
    ca = ChannelAttention(64).cuda()
    sa = SpatialAttention().cuda()
    
    out=ca(out)*out
    out=sa(out)*out

    model1=nn.Conv2d(64,3,kernel_size=3,padding=1).cuda()
    out=model1(out)
    print(out.shape)
    # visiualize heatmap about ccAttention
    out = out.cpu().squeeze(0).detach().numpy()
    out = mask_heatImage(Test_Img, out)
    cv2.imshow("result", out)
    cv2.waitKey(0)