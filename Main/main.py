'''
Author: AKiy github:Akiy071
Date: 2024-03-07 16:56:16
LastEditors: AKiy
LastEditTime: 2024-05-02 09:09:35
Description: 
'''
import os
import argparse
import random
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torchvision.utils import make_grid

import matplotlib.cm as cm
from matplotlib.colors import Normalize

import sys
sys.path.append(r"DataProcess")


from Model.YOLOV5_only import Segment
from Model.Unet_only import MyNet
from Loss.MyLoss import Myloss
from DataProcess.DataSet import MyDataSet

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 只显示 warning 和 Error

def get_neighbor(images,rows,cols,center_row,center_col):
    # Get Center img patch neighbor patch
    lb_res=[]
    img_lst=[]
    for i in range(center_row-1,center_row+2):
        for j in range(center_col-1,center_col+2):
            if i<0 or i>rows-1:
                continue
            if j<0 or j>cols-1:
                continue
            
            label=torch.from_numpy(images[i][j])
            label=torch.where(label[2]>0,1,0)
            
            lb_res.append(label)
            img_lst.append((i,j))
            
    return lb_res,img_lst

def collate_fn(batch):
    img=[]
    lb=[]
    img_lst=[]
    rows,cols=0,0
    # get patch [p,3,patch_size,patch_size]
    for k in range(len(batch)):
        images,labels=batch[k]
        for i in range(len(images)):
            rows=len(images)
            for j in range(len(images[i])):
                cols=len(images[i])
                label=torch.from_numpy(labels[i][j])
                label=torch.where(label[2]>0,1,0)
                if len(torch.unique(label))>1:
                    lb_neighbor,img_index=get_neighbor(labels,rows,cols,i,j)
                    lb+=lb_neighbor
                    img_lst+=img_index
            # Get image by index.
        for n in range(len(img_lst)):
            image=torch.from_numpy(images[img_lst[n][0]][img_lst[n][1]])
            img.append(image)

    return torch.stack(img),torch.stack(lb)

def initalize_model(model):
    for name,param in model.named_parameters():
        if "weight" in name and param.data.ndim >=2:
            nn.init.xavier_uniform_(param.data)
            if "relu" in name.lower():
                nn.init.kaiming_uniform_(param.data,a=0,mode="fan_in",nonlinearity="relu")
        elif "bias" in name:
            nn.init.constant_(param.data,0)

# Using Hook function to get forward feature map.
conv_fmap_ls = []
def hook_function(model,input,output):
    conv_fmap_ls.append(output) 

def Show_featuremap(conv_fmap_ls,writer,epoch,nrow=8):
    if len(conv_fmap_ls)>0:
        fm=conv_fmap_ls[0]
        fm=fm.unsqueeze(2)
        b,output_c,c,w,h=fm.size()
        fm=fm.view(-1,c,w,h)
        
        norm = Normalize(vmin=fm.min(), vmax=fm.max())
        
        # Apply colormap 
        # blue mean grayscale approch 0,red approch 1,green mean medium
        colormap = cm.get_cmap("brg",8)
        fm_mapped = torch.cat([torch.tensor(colormap(norm(fm[i].cpu().detach().numpy()))) for i in range(fm.size(0))], dim=0).permute(0,3,1,2)

        gm=make_grid(fm_mapped,nrow=nrow,normalize=False)
        writer.add_image("attention1_feature_map",gm,epoch)
            
        conv_fmap_ls.clear()
        
if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir", type=str, help="image directory path",
                        default=r"Data\BSData-main\Data_Split_clear")
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

    ############################################################################
    # Set random seed
    ############################################################################
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    ############################################################################
    #GPU or CPU
    ############################################################################

    if args.GPU and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print("Model Training Will Use {} GPU".format(args.GPU))
    else:
        device = torch.device("cpu")
        print("Model Training will Use CPU, Please Be Attention Your Training Time.")
        
    ############################################################################
    #Define Model & initalize model weight & choice whether keeping train.
    ############################################################################

    #model = Segment(3,32,2).to(device)
    model=MyNet(3,2).to(device)
    
   
        
    if args.keep_train:
        state_dict=torch.load(args.model_path+r"\2024-04-08_23-05\epoch_790_model.0.0003_YOLO-only.t7")
        model.load_state_dict(state_dict)
        print("It will keep using last time model weights file to train.")
    else:
        # Model weight initalized.
        initalize_model(model)
        print("Using xavir to initalize model weights.")

    ############################################################################
    #Loss Function & Optimizer & lr scheduler 
    ############################################################################

    #This Part is for downing the backgorund weight in loss.
    #_______________________________________________________
    weights = torch.ones([2]).to(device)*1 # Number of categories 
    for w in weights:
        w=10
    weights[0] =0.1

    criterion=Myloss(weights=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))
    scheduler = CosineAnnealingLR(optimizer,T_max=20, eta_min=0.0015) 
    ############################################################################
    #ImageDaset Loader
    ############################################################################
    
    Trainset=MyDataSet(args,"train")
    Valset=MyDataSet(args,"test")
    train_iter = DataLoader(Trainset, batch_size=1,
                            shuffle=True, drop_last=True, num_workers=1,collate_fn=collate_fn)
    val_iter = DataLoader(Valset, batch_size=1,
                            shuffle=True, drop_last=True, num_workers=1,collate_fn=collate_fn)

    ############################################################################
    #visualize the training process
    ############################################################################

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    folder_name=args.model_path+f"/{current_time}" # model save path
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    writer=SummaryWriter(log_dir=args.log_dir+f"/{current_time}") # trianing log.

    epochs=args.epochs
    pbar=tqdm(total=epochs)

    #Begin train
    best_test_loss=np.Inf
    for epoch in range(epochs):
        pbar.update(1)
        pbar.set_description("Current epoch：{}".format(epoch+1))
        
        epoch_loss = 0.0
        epoch_acc = 0.0
        
        train_loss, train_acc = 0.0, 0.0
        val_loss, val_acc = 0.0, 0.0
        
        for phase in ["train", "val"]:
            if phase=="train":
                model.train()
                
                 # add hook function to watch forward feature map.
                # if args.is_hook:
                #     hook=model.output.register_forward_hook(hook_function)
                    
            else:
                model.eval()
            run_loss = 0.0
            run_acc = 0.0
            count = 0
            for images, labels in (train_iter if phase == "train" else val_iter):
                images, labels = images.to(device), labels.to(device)
                batch_size = images.shape[0]
                count += batch_size
                
                with torch.set_grad_enabled(phase=="train"):
                    optimizer.zero_grad()
                    logits=model(images).to(device)

                    loss=criterion(logits,labels.long())
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    run_loss += loss.item()*batch_size
                    run_acc += torch.sum((torch.argmax(logits.data, 1)) ==
                                            labels.data) / (args.patch_size[0] * args.patch_size[1])
            
            epoch_loss = run_loss / count
            epoch_acc = run_acc.double() / count
            if phase == "train":
                train_loss = epoch_loss
                train_acc = epoch_acc
            else:
                val_loss = epoch_loss
                val_acc = epoch_acc
                
            print("{}/Epoch_Loss: {:.4f} Epochs_ACC: {:.4f}".format(phase,
                    epoch_loss, epoch_acc))
            
            # save model file.
            if epoch_loss<best_test_loss and phase=="val" and epoch&10!=0:
                best_test_loss=epoch_loss 
                if os.path.exists(folder_name+"/best_YOLO-CBAM.t7"):
                    os.remove(folder_name+"/best_YOLO-CBAM.t7")
                torch.save(model.state_dict(),
                        folder_name+"/best_YOLO-CBAM.t7")
            
            if epoch%10==0 and phase=="val":
                if os.path.exists(folder_name+"/last_YOLO-CBAM.t7"):
                    os.remove(folder_name+"/last_YOLO-CBAM.t7")
                torch.save(model.state_dict(),
                        folder_name+"/last_YOLO-CBAM.t7")
        
        LR=optimizer.param_groups[0]["lr"]
        print("epoch:%d Learning_rate is ：%.6f"%(epoch,LR))
        
        # record feature map
        # Show_featuremap(conv_fmap_ls,writer,epoch,16)
        
        # hook.remove()
        
        writer.add_scalars(
                "Loss", {"Train_Loss": train_loss, "Val_loss": val_loss}, epoch)
        writer.add_scalars(
            "Pixel_Accuracy", {"Train_Acc": train_acc, "Val_Acc": val_acc}, epoch)
        writer.add_scalar(
            "Learning_Rate",LR,epoch)
        
        pbar.refresh()
        scheduler.step()
        
    writer.close()