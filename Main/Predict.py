'''
Author: AKiy github:Akiy071
Date: 2024-03-08 13:06:14
LastEditors: AKiy
LastEditTime: 2024-05-03 09:42:54
Description: 
'''
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
import argparse
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
import time
from tqdm import tqdm

from Model.Unet_only import MyNet
from Model.YOLOV5_only import Segment
from DataProcess.DataSet import MyDataSet
from Loss import MyEvaluation


parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", type=str, help="image directory path",
                default=r"Data\BSData-main\Data_Split")
parser.add_argument("--out_path", type=str,
                help="output directory path", default="./DataProcess/output/")
parser.add_argument("--patch_size", type=tuple,
                default=(256, 256), help="The Crop size decide on you.")
parser.add_argument("--model_path", type=str,
                    default=r"Z:\Desktop\python项目\03_BEiTwithAttention\OutPut\Unet-only\epoch_300_model.0.2520.t7")
args = parser.parse_args()

def Image_Concat(images,beta=0.2):
    # Please ensure this beta value is Same as DataSet.py._cut_image function.
    rows,cols=len(images),len(images[0])
    
    width_stride=int(args.patch_size[1]*(1-beta))
    height_stride=int(args.patch_size[0]*(1-beta))
    
    w_diff=args.patch_size[1]-width_stride
    h_diff=args.patch_size[0]-height_stride
    
    pad_height=args.patch_size[0]-(rows*height_stride+h_diff)%args.patch_size[0] # 填充行数
    pad_width=args.patch_size[1]-(cols*width_stride+w_diff)%args.patch_size[1] # 填充列数
    
    images_concat=torch.zeros(3,rows*width_stride+(args.patch_size[0]-width_stride),cols*height_stride+(args.patch_size[1]-height_stride))
    
    images_concat=np.pad(images_concat,((0,0),(pad_height//2,pad_height//2),(pad_width//2,pad_width//2)),mode="constant",constant_values=0)
    images_concat=torch.tensor(images_concat)
    
    # images_concat=torch.zeros(3,rows*width_stride+(args.patch_size[0]-width_stride),cols*height_stride+(args.patch_size[1]-height_stride))
    for i in range(rows):
        for j in range(cols):
            images_concat[:,i*height_stride:i*height_stride+args.patch_size[0],j*width_stride:j*width_stride+args.patch_size[1]]=images[i][j].squeeze(0)
    
    images_concat=images_concat[:,pad_height//2:-pad_height,75:-75]
    
    return images_concat  


def Show(image,label,pre):
    image=Image_Concat(image)
    mean=[0.485,0.456,0.406]
    std=[0.229,0.224,0.225]
    image=image.cpu().numpy().transpose(1,2,0)
    image=((image*std+mean)*255).astype("uint8")
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    label=label.astype(np.uint8)
    # label=cv2.cvtColor(label.transpose(1,2,0),cv2.COLOR_BGR2RGB)
    lst_name=["image","label","pre"]
    lst=[image,label,pre]
    
    for i in range(len(lst)):
        plt.subplot(3,1,i+1)
        plt.imshow(lst[0])
        if i!=0:
            plt.imshow(lst[i],alpha=0.6)
        plt.title(lst_name[i])
    
    plt.show()     


def Model_test1():
    model_state=torch.load(args.model_path)
    model=MyNet(3,2).cuda()
    #model=Segment(3,32,2).cuda()
    
    model.load_state_dict(model_state)
    model.eval()

    datasets=MyDataSet(args,"test") # train test
    image_names=datasets.Dataloader.Image_name
    
    val_iter = DataLoader(datasets, batch_size=1,
                            shuffle=False, drop_last=True, num_workers=1)
    
    with torch.no_grad():
        PA,mIou,mPA,MSR=[],[],[],[]
        P,R=[],[]
        metric=MyEvaluation.MyEvaluation()
        num=0
        for images,labels in tqdm(val_iter):
            preds=[]
            for i in range(len(images)):
                images_input=torch.stack(images[i],dim=1).squeeze(0).cuda()
                out=model(images_input)
                out=torch.softmax(out,dim=1)
                out=torch.argmax(out,dim=1)
                preds.append(list(out))
            label=Image_Concat(labels)
            preds=Image_Concat(preds)
            
            label=label.cpu().numpy().transpose(1,2,0)
            Save_Map(images,label,preds,image_names[num]) #write for check out model effection specifically.
            num+=1
            preds=preds.cpu().numpy().transpose(1,2,0)
            metric.GetConfusionMatrix(metric,label,preds,2)
            
            MSR.append(metric.MSR())
            mIou.append(metric.mIou())
            mPA.append(metric.mPA())
            PA.append(metric.pixelAccuracy())
            P.append(metric.P_R()[0])
            R.append(metric.P_R()[1])
            #Show(images,label,preds) # When you need to evaluate model,please Cancel this fuction.
        
    print("-------------------------------------------------------")
    print("mIou:{:.4f},mPA:{:.4f},MSR:{:.4f}".format(sum(mIou)/len(mIou),sum(mPA)/len(mPA),sum(MSR)/len(MSR)))
    print("PA:{:.4f}".format(sum(PA)/len(PA)))
    print("P:{:.4f},R:{:.4f}".format(sum(P)/len(P),sum(R)/len(R)))

def Save_Map(images,label,pred,image_names,category_id=1,save_path="./ConcatMap"):
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    image_names=image_names.split(".")[0]
    mask_uint8=255*np.uint8(pred==category_id)
    
    #Image concat and save
    image=Image_Concat(images)
    mean=[0.485,0.456,0.406]
    std=[0.229,0.224,0.225]
    image=image.cpu().numpy().transpose(1,2,0)
    image=((image*std+mean)*255).astype("uint8")
    spacing = np.ones((image.shape[0], 5, 3), np.uint8) * 255 
    all_image=np.hstack((image,spacing,label,spacing,mask_uint8.transpose(1,2,0)))
    cv2.imwrite(save_path+"/"+image_names+".jpg",all_image)
        
if __name__=="__main__":
    Model_test1()