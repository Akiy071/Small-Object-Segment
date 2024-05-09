import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax
from torchsummary import summary

def MaskAttention(B, H, W):
    """
    生成斜对角线为无穷的三维矩阵，大小为（B*W,H,H）
    """
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class RCC_Attention(nn.Module):

    def __init__(self, in_dim):
        super(RCC_Attention, self).__init__()
        self.Query_Conv = nn.Conv2d(in_dim, in_dim//3, kernel_size=1)
        self.Key_Conv = nn.Conv2d(in_dim, in_dim//3, kernel_size=1)
        self.Value_Conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.Norm = nn.Softmax(dim=3)
        self.gama = nn.Parameter(torch.zeros(1))
        self.INF = MaskAttention

    def forward(self, x):
        Batch_size, C, H, W = x.size()  # 获取图片大小信息

        # 获取Q、K、V
        Q = self.Query_Conv(x)
        K = self.Key_Conv(x)
        V = self.Value_Conv(x)

        Q_H = Q.permute(0, 3, 1, 2).contiguous().view(
            Batch_size*W, -1,H).permute(0,2,1)
        Q_W = Q.permute(0, 2, 1, 3).contiguous().view(
            Batch_size*H,-1,W).permute(0,2,1)

        K_H = K.permute(0, 3, 1, 2).contiguous().view(Batch_size*W, -1, H)
        K_W = K.permute(0, 2, 1, 3).contiguous().view(Batch_size*H, -1, W)

        V_H = V.permute(0, 3, 1, 2).contiguous().view(Batch_size*W, -1, H)
        V_W = V.permute(0, 2, 1, 3).contiguous().view(Batch_size*H, -1, W)

        # 点乘
        P_H = torch.bmm(Q_H, K_H).cuda()  # 得到相似度
        P_H += self.INF(Batch_size, H, W)
        P_H=P_H.view(Batch_size, W, H, H).permute(0, 2, 1, 3)
        P_W = torch.bmm(Q_W, K_W).cuda()
        P_W = P_W.view(Batch_size, H, W, W)

        # 归一化，得到权重矩阵
        concate = self.Norm(torch.cat([P_H, P_W], 3))
        # del P_W,P_H
        # torch.cuda.empty_cache()
        
        att_W = concate[:, :, :, H:H+W].contiguous().view(Batch_size*H, W, W)
        att_H = concate[:, :, :, 0:H].permute(
            0, 2, 1, 3).contiguous().view(Batch_size*W, H, H)
        # del concate
        # torch.cuda.empty_cache()

        output_H = torch.bmm(V_H, att_H.permute(0, 2, 1))
        output_H = output_H.view(
            Batch_size, W, -1, H).permute(0, 2, 3, 1)  # (b,c1,h,w)
        # del att_H
        # torch.cuda.empty_cache()
        
        output_W = torch.bmm(V_W, att_W.permute(0, 2, 1))
        output_W = output_W.view(
            Batch_size, H, -1, W).permute(0, 2, 1, 3)  # (b,c1,h,w)
        # del att_W
        # torch.cuda.empty_cache()
        
        return self.gama*(output_H+output_W)+x  # 加权求和


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
    # print(type(Test_Img))
    # cv2.imshow("Img", Test_Img)
    # cv2.waitKey(0)
    x = torch.tensor(Test_Img).unsqueeze(0).permute(0, 3, 1, 2).float().cuda()
    print(x.shape)
    model = RCC_Attention(3).cuda()
    x = nn.Conv2d(3, 3, kernel_size=1).cuda()(x)
    out = model(x)
    print(out.shape)
    # summary(model, (3, 280,560))
    # out = nn.Conv2d(512, 3, kernel_size=1)(out)

    # visiualize heatmap about ccAttention
   #  out = out.squeeze(0).detach().numpy()
   #  out = mask_heatImage(Test_Img, out)
   #  cv2.imshow("result", out)
   #  cv2.waitKey(0)