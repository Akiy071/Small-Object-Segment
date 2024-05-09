'''
Author: AKiy github:Akiy071
Date: 2024-03-25 14:13:29
LastEditors: AKiy
LastEditTime: 2024-05-03 09:38:26
Description: 
'''
import numpy as np
import sys
sys.path.append(r"Z:\Desktop\python项目\03_BEiTwithAttention")
from Predict import Image_Concat

class MyEvaluation():
    def __init__(self,grounds=None,preds=None,class_num=2,) -> None:
        self.grounds,self.preds=grounds,preds
        self.class_num=class_num
        
        if self.grounds!=None and preds!=None:
            self.confusionmatrix=self._GetConfusionMatrix(self.class_num) # Initializing confusion matrix
            
    @staticmethod
    def GetConfusionMatrix(metric,grounds,preds,class_num):
        grounds=np.where(grounds>1,1,0)
        mask=(grounds>=0)&(grounds<class_num) # Get mask matrix
        label=class_num*grounds[mask]+preds[mask]
        count = np.bincount(label.astype(int), minlength=class_num**2)
        confusionMatrix = count.reshape(class_num, class_num)
        metric.confusionmatrix=confusionMatrix
    
    def _GetConfusionMatrix(self):
        mask=(self.grounds>=0)&(self.grounds<self.class_num) # Get mask matrix
        label=self.class_num*self.grounds[mask]+self.preds[mask]
        count = np.bincount(label, minlength=self.class_num**2)
        confusionMatrix = count.reshape(self.class_num, self.class_num)
        return confusionMatrix
    
    def mIou(self):
        # Iou= TP/(TP+FP+FN)
        # mIou=[Iou+TN/(TN+FP+FN)]/class_num
        intersection=np.diag(self.confusionmatrix)
        union=np.sum(self.confusionmatrix,axis=1)+np.sum(self.confusionmatrix,axis=0)-intersection
        mIou=np.nanmean(intersection/union)
        return mIou
    
    def CIou(self):
        #CIou=TN/(TN+FP+FN)
        TN=self.confusionmatrix[0][1]
        FP_FN=self.confusionmatrix[0][1]+self.confusionmatrix[1][0]
        CIou=TN/(TN+FP_FN)
        return CIou
        
    def Dice(self):
        TP=self.confusionmatrix[0][0]
        FP_FN=self.confusionmatrix[0][1]+self.confusionmatrix[1][0]
        DSC=2*TP/(FP_FN+2*TP)
        return DSC
    
    def pixelAccuracy(self):
        # PA=(TP+TN)/(TP+TN+FP+TN)
        acc=np.diag(self.confusionmatrix).sum()/self.confusionmatrix.sum()
        return acc
    
    def ClassPixelAccuracy(self):
        # class_acc=(TP)/TP+FP or TN/(TN+FN)
        classAcc=np.diag(self.confusionmatrix)/self.confusionmatrix.sum(axis=1)
        return classAcc
    
    def mPA(self):
        classAcc=self.ClassPixelAccuracy()
        meanAcc=np.nanmean(classAcc)
        return meanAcc
    
    def FAR(self):
        # FAR=FN/(FN+TP)
        return np.diag(self.confusionmatrix)[1]/(self.confusionmatrix.sum(axis=1)[0])
    
    def MSR(self):
        # MSR=FP/(TN+FP)
        return float(np.diag(self.confusionmatrix,k=-1)/(self.confusionmatrix.sum(axis=1)[1]+1e-10))
    
    def P_R(self):
        
        TP=self.confusionmatrix[0][0]
        TP_FP=TP+self.confusionmatrix[1][0]
        TP_FN=TP+self.confusionmatrix[0][1]
        
        return [TP/TP_FP,TP/TP_FN]
