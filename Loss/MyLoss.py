'''
Author: AKiy github:Akiy071
Date: 2024-01-11 13:56:21
LastEditors: AKiy
LastEditTime: 2024-04-16 21:41:20
Description: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Myloss(nn.Module):
    def __init__(self, weights=None):
        super(Myloss, self).__init__()
        self.weights = weights

    def CE_loss(self, preds, labels):
        if self.weights is None:
            logits = torch.nn.functional.softmax(preds.float(), dim=1)

            logits_log = torch.log(logits)
            loss = F.nll_loss(logits_log, labels.long())
        else:
            loss = F.cross_entropy(
                input=preds, target=labels, weight=self.weights)
        return loss

    def Dice_loss(self, preds, target, smooth=1e-5):
        """
        :param pred: [b,num_classes,h,w]  where num_class is the probability that a pixel belongs to that class.
        :param target: [b,h,w]
        """
        sum_dice = 0
        num_class = preds.shape[1]
        
        if num_class>2:preds = F.softmax(preds,dim=1)
        else:preds = F.sigmoid(preds) # if multi-class change it to softmax.
        
        preds = torch.argmax(preds, dim=1)
        for i in range(num_class):
            
            preds = preds.contiguous().view(preds.shape[0], -1)  # 数据展平
            target = target.contiguous().view(target.shape[0], -1)  # 数据展平

            intersection = (preds * target).sum()
            dice = (2. * intersection + smooth) / \
                (target.sum() + preds.sum() + smooth)
            sum_dice += dice
        return 1 - sum_dice / num_class

    def BCE_loss(self, preds, target,epsilon=1e-10):
        num_classes=preds.shape[1]
        preds = torch.sigmoid(preds)
        target=F.one_hot(target.to(torch.int64),num_classes).permute(0,3,1,2)
        
        loss=torch.mean(-(target*torch.log(preds+epsilon)+(1-target)*torch.log((1-preds+epsilon)))*self.weights.unsqueeze(1))
        return loss
    
    def Focal_Loss(self, pred_logit, target, alpha_pos=0.20, alpha_neg=0.80, gamma=2):
        B,C=pred_logit.shape[:2] # Batch size and Number of Categories
        
        pred_logit=pred_logit.reshape(B,C,-1) # flatten
        pred_logit=pred_logit.transpose(1,2) # [B,H*W,C]
        pred_logit=pred_logit.reshape(-1,C) # [B*H*W,C]
        target=target.reshape(-1) #[B*H*W]
        
        log_p=torch.log_softmax(pred_logit,dim=-1)
        log_p=log_p.gather(1,target[:,None]).squeeze() # [B*H*W,]
        p=torch.exp(log_p)
        
        alpha = torch.where(target == 1, torch.tensor(alpha_pos, dtype=torch.float, device=pred_logit.device), 
                         torch.tensor(alpha_neg, dtype=torch.float, device=pred_logit.device))
        
        
        loss=-1*alpha*torch.pow(1-p,gamma)*log_p
        return loss.sum()/target.numel()
    
    def forward(self, preds, labels):
        self.weights=torch.zeros_like(labels).float()
        if self.weights!=None:
             # Weights matrix 
            self.weights=torch.fill_(self.weights,0.25)
            self.weights[labels>0]=2
        else:
            self.weights=torch.fill_(self.weights,1)
        loss = self.Dice_loss(preds,labels)
        loss += self.Focal_Loss(preds,labels)
        return loss

    def IouLoss(self,pred_logit,target,smooth=1e-10):
        log_p=torch.log_softmax(pred_logit,dim=1)
        log_p=torch.argmax(log_p,dim=1)
        intersection=log_p*target
        loss=(intersection.sum()+smooth)/(log_p.sum()+target.sum()-intersection.sum()+smooth)
        
        return 1-loss.mean()


if __name__=="__main__":
    pred=torch.tensor([[ [[0.88,0.78,0.98,0.88],[0.96,0.97,0.98,0.98],[0.78,0.01,0.94,0.79],[0.75,0.79,0.95,0.93]],
                     [[0.01,0.03,0.02,0.02],[0.05,0.98,0.09,0.07],[0,0.88,0.20,0.12],[0,0.1,0.0,0.12]] ]])
    pred=torch.tensor([[ [[1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0],[1.0,0,0,1.0],[1.0,1.0,1.0,1.0]],[[0,0,0,0],[0,0,0,0],[0,1.0,1.0,0],[0,0,0,0]]]])
    label=torch.tensor([[[0,0,0,0],[0,0,0,0],[0,1,1,0],[0,0,0,0]]])
    weights = torch.ones([2]).cuda()*1 # Number of categories 
    for w in weights:
        w=10
    weights[0] =0.1
    criterion=Myloss(None)
    output=criterion(pred,label)
    output1=criterion.IouLoss(pred,label)
    print(output)
        