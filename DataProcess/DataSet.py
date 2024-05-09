'''
Author: AKiy github:Akiy071
Date: 2024-03-07 13:10:26
LastEditors: AKiy
LastEditTime: 2024-04-16 22:07:49
Description: 
'''
import numpy as np
import torch
import argparse
import os
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from Data_Process import DataFileTypeTo
"""
Processing Image and Label.
"""

class Read_Data():
    
    def __init__(self,args) -> None:
        self.args=args
        self.image_dir=args.image_dir
        self.Lab_out_Path=args.out_path #save .png Label image.
        self.ImageTransformer=DataFileTypeTo(args)
        self.Image_name=[]
        #if not transform json file to png.
        if not os.path.exists(self.Lab_out_Path):
            self.ImageTransformer.json2png()
        

    def Load_Img(self,method="train"):
        Images,Labels=[],[]
        for dir_name in os.listdir(self.Lab_out_Path):
            if dir_name != method:
                continue
            file_dir=os.path.join(self.Lab_out_Path,dir_name)
            
            count=0
            count+=(len(os.listdir(file_dir))-1)*2
            
            if count!=0:
                pbar=tqdm(total=count)
                pbar.set_description(file_dir+" is Load:")
                
                for file_name in os.listdir(file_dir):
                    if file_name.endswith(".jpg"):
                        image_name = os.path.join(file_dir, file_name)
                        if not os.path.isfile(image_name):
                            return print("The Path {0} doesn't exist {1}").format(file_dir, file_name)
                        imageData = cv2.imread(image_name)  # H,W,C
                        
                        self.Image_name.append(file_name)
                        
                        Images.append(imageData) #Train image Loaded.
                        pbar.update(1)    

                    if file_name=="Label":
                        lb_file_dir=os.path.join(file_dir,file_name)
                        for lb_file_name in os.listdir(lb_file_dir):
                            lb_img_name=os.path.join(lb_file_dir,lb_file_name)
                            if not os.path.isfile(lb_img_name):
                                return print("The Path {0} doesn't exist {1}").format(lb_file_dir, lb_img_name)
                            lb_Data=cv2.imread(lb_img_name) # H,W,C
                            Labels.append(lb_Data)
                            pbar.update(1)
                pbar.close()
        return Images,Labels

class MyDataSet(Dataset):
    def __init__(self, args,method="train") -> None:
        super().__init__()
        self.args = args
        self.patch_size = args.patch_size
        self.Dataloader=Read_Data(args)
        self.images, self.labels = self.Dataloader.Load_Img(method)
                    
        # Numpy to Tensor
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225]),
        ])

    def get_image_name(self):
        
        return self.Dataloader.Image_name
        
    def _cut_image(self,image,patch_size,beta=0.2):

        height=image.shape[1]
        width=image.shape[2]
        
        pad_height=patch_size[0]-height%patch_size[0] # 填充行数
        pad_width=patch_size[1]-width%patch_size[1] # 填充行数
        
        pad_image=np.pad(image,((0,0),(pad_height//2,pad_height//2),(pad_width//2,pad_width//2)),mode="constant",constant_values=0)
        # Split
        patches=[]
        width_stride=int(patch_size[0]*(1-beta))
        height_stride=int(patch_size[1]*(1-beta))

        for i in range(0,pad_image.shape[1]-patch_size[0],height_stride):
            w_patches=[] # store image cols.
            for j in range(0,pad_image.shape[2]-patch_size[1],width_stride):            
                patch=pad_image[:,i:i+patch_size[0],j:j+patch_size[1]]
                w_patches.append(patch)
            patches.append(w_patches)
        return patches
    
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = self.transformer(image)  # C H W
        label = torch.from_numpy(np.asarray(label, dtype=np.int32)).permute(2,0,1)
        image_patches = self._cut_image(image,self.patch_size)
        label_patches = self._cut_image(label,self.patch_size)
        return image_patches, label_patches

    def __len__(self):
        return len(self.images)


if __name__=="__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir", type=str, help="image directory path",
                        default=r"Data\BSData-main\Data_Split")
    parser.add_argument("--out_path", type=str,
                        help="output directory path", default="./DataProcess/output/")
    parser.add_argument("--patch_size", type=tuple,
                        default=(256, 256), help="The Crop size decide on you.")

    #train_config
    parser.add_argument("--epochs",type=int,default=600)

    parser.add_argument("--GPU",type=int,default=1,help="which number is Using GPU to Train.Default:1")
    parser.add_argument("--keep_train",type=bool,default=False,help="Whether keep training with last model")
    parser.add_argument("--model_path",type=str,default="./OutPut",help="The Path of your model saved.")
    parser.add_argument("--log_dir",type=str,default="./Log",help="Where your decide to Save train log file.")
    parser.add_argument("--is_hook",type=bool,default=False,help="Get model any layer feature map to show.")
    
    parser.add_argument("--seed",type=int,default=42,help="Your lucky number.")

    args = parser.parse_args()
    Trainset=MyDataSet(args,"test")
    train_iter = DataLoader(Trainset, batch_size=1,shuffle=True, drop_last=True, num_workers=1)
    
    for images,labels in train_iter:
        print(images.shape)
        print(labels.shape)