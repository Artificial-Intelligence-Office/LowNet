#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:23:04 2020

@author: luu
"""


import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
from PIL import Image
import copy
import time
from torch.utils.data import Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import shutil
import random
resize=16
size=16
sample=1
slopes=[[0.25,0.25,0.25],[1,1,1],[4,4,4],[0.25,1,4],
        [1,2,4],[1,4,8],[8,4,1],[4,2,1]]


transform=torchvision.transforms.ToTensor()
transform_train=torchvision.transforms.ToTensor()





'''
transform_train=torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomAffine((-10,10)),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.Pad(2, fill=0, padding_mode='constant'),
        torchvision.transforms.RandomCrop(16),
        torchvision.transforms.ToTensor(),
        ])
'''
device='cuda:0'
class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors)
        self.tensors = tensors
        self.transform = transform
    def __getitem__(self, index):
        x = self.tensors[0][index]
        if self.transform:
            x = self.transform(x)
        #z = torch.from_numpy(self.tensors[1][index])
        z = self.tensors[1][index] 
        return x, z
    def __len__(self):
        return self.tensors[0].shape[0] 



def ckld_loss(y_pred,y_true, alpha=0.5,gamma=1,epsilon=1e-10):


    y_true = torch.clamp(y_true, epsilon, 1)
    y_pred = torch.clamp(y_pred, epsilon, 1)
    return torch.sum(torch.mul(alpha * torch.mul(y_true , (1-y_pred)**gamma),  (torch.log(y_true)- torch.log(y_pred))))
#    return torch.sum(torch.mul(alpha * torch.mul(y_true , (1-y_pred)**gamma),  (torch.log(y_true)- y_pred)))
def focal_loss(y_pred,y_true,gamma=2,epsilon=1e-10):

    y_true = torch.clamp(y_true, epsilon, 1)
    y_pred = torch.clamp(y_pred, epsilon, 1)
    return torch.sum( torch.mul(-1*y_true, torch.mul((1-y_pred)**gamma,torch.log(y_pred))))

def new_relu(y,slope=1):
    return F.relu(y)*slope

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
def plot():
    plt.figure()
    
    plt.plot(train_acc_new)
    plt.plot(test_acc_new)
    
    plt.title('model accuracy_'+str(slope[0])+'_'+str(slope[1])+'_'+str(slope[2])+'___'+str(i))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.savefig(dest_path_new+'Accuracy_'+tail+'.png')
    #plt.show()
    # summarize history for loss
    plt.figure()
    
    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(dest_path_new+'Loss_'+tail+'.png')
    
    #plt.show()
    hist={'train_loss':train_losses,'test_loss':test_losses,'train_acc':train_acc_new,'test_acc':test_acc_new}
    df = pd.DataFrame(hist)
    df.to_csv(dest_path_new+'/history_'+tail+'.csv', index=False)   

class Net(nn.Module):
    def __init__(self,slopes=[2,1,0.5]):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        #self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        #self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        #self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.drop_1 = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(100*128, 128)
        #self.fc1 = nn.Linear(256*128,128)
        self.drop_2 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.slopes=slopes
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = new_relu(self.conv1(x),slope=self.slopes[0])
        x = new_relu(self.conv2(x), slope=self.slopes[1])
        x = new_relu(self.conv3(x), slope=self.slopes[2])
        #x = F.relu(self.conv1(x))
        #x = F.relu(self.conv2(x))
        #x = F.relu(self.conv3(x))
        x = F.dropout(x, p=0.3)        
        x= x.view(x.size(0), -1)
        #x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x= F.dropout(x,p=0.3)
        x = F.relu(self.fc2(x))
        x= F.dropout(x,p=0.3)
        x = F.relu(self.fc3(x))
        return self.softmax(x) 
        #return F.log_softmax(x,-1)
def get_prediction(model):
    model.eval()
    y_test_eval=[]
    y_pred=[]
    for inputs , labels in dataloaders['testing']:
        inputs = inputs.type('torch.FloatTensor').to(device)
        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        for i in labels.detach().numpy():
            y_test_eval.append(i )
        for i in preds.cpu().detach().numpy():
            y_pred.append(i)
    return y_test_eval, y_pred
    

def train(model, criterion, optimizer, num_epochs=250):
    model.train()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc=0.0
    best_epoch=0
    since=time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)        

        # Each epoch has a training and validation phase
        for phase in [ 'training','testing']:
            if phase == 'training':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            r=0
            # Iterate over data.
            for inputs , labels in dataloaders[phase]:
                r=r+1
                inputs = inputs.type('torch.FloatTensor').to(device)
                #labels = labels.type('torch.LongTensor')
                #labels = labels.type('torch.LongTensor').to(device)
                labels= labels.type('torch.LongTensor')
                #labels = labels.type('torch.FloatTensor').to(device)
                targets=torch.zeros(len(labels), num_classes).scatter_(1, labels.unsqueeze(1), 1.).to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'training'):
                    
                    outputs = model(inputs)
                    #print(outputs.shape)
                    _, preds = torch.max(outputs, 1)
                    #loss=criterion(outputs,labels)
                    loss = criterion(outputs, targets)
                    
                    # backward + optimize only if in training phase
                    if phase == 'training':
                        loss.backward()
                        optimizer.step()
                #running_loss += loss.item() * inputs.size(0)
                running_loss += loss.item() 
                #print('r:',loss.item()) 
                running_corrects += torch.sum(preds == labels.to(device).data)
            #print('rr:',running_loss,dataset_sizes[phase])    
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase=='training':
                train_losses.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:                
                test_losses.append(epoch_loss)
                test_acc.append(epoch_acc)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss,epoch_acc, epoch))
            
            # deep copy the model
            if phase == 'testing' and epoch_acc > best_acc:
                    best_epoch=epoch
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())



    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best epoch: {:4f}'.format(best_epoch))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

conf=19
i=0
learning_rate=0.001
n_epochs=100
log_interval=10

path='Datas/poses_'+str(size)+'x'+str(size)+'_reduced_splitted_balanced_dataset'

data={}
names=['x_training_','y_training_','x_testing_','y_testing_']
#for fold in range(1):
fold=0
for k in names:
    data[k+str(fold)]=[]

    #class_index=pd.read_csv('Class_reduction.csv')
#for fold in range(1):

for k in ['testing','training']:
    classes=os.listdir(os.path.join(path,k))
    classes.sort(key=int)
    for clas in classes:
        if True:
            temp_path=os.path.join(path,k,clas)
            imgs=os.listdir(temp_path)
            #print(len(imgs))
            for img in imgs :
                data['x_'+k+'_'+str(fold)].append(np.asarray(Image.open(os.path.join(temp_path,img))))#.resize((resize,resize))))
                data['y_'+k+'_'+str(fold)].append(int(clas))   

#for i in range(1):s

for k in names:
    data[k+str(fold)]=np.asarray(data[k+str(fold)])

batch_size = 1024
num_classes = max(data['y_training_'+str(fold)])+1
epochs = 12

# input image dimensions
img_rows, img_cols = resize,resize

# the data, split between train and test sets


fold=0
(x_train, y_train), (x_test, y_test) = (data['x_training_'+str(fold)],data['y_training_'+str(fold)]),(data['x_testing_'+str(fold)],data['y_testing_'+str(fold)])
del(data)

Datas={'training':CustomTensorDataset((x_train, y_train),transform=transform_train),
       'testing':CustomTensorDataset((x_test, y_test),transform=transform)}

dataset_sizes= {x : len(Datas[x]) for x in ['training','testing']}

dataloaders= {x : torch.utils.data.DataLoader(Datas[x], batch_size=batch_size, shuffle=True)
                for x in ['training', 'testing']}


dest_path='Results/Results_cleaned_reduced_classes_16x16_pytorch_splitted_slopes_focal_loss/'
slopes=[[4,2,1]]
for slope in slopes:
    dest_path_new=dest_path+'slopes/'+str(slope[0])+'_'+str(slope[1])+'_'+str(slope[2])+'/'
    #if os.path.exists(dest_path_new):
    #    shutil.rmtree(dest_path_new)
    createFolder(dest_path_new)
    
    for i in range(9):
        
        train_losses = []
        test_losses = []
        train_acc= []
        test_acc=[]
        #loss=nn.KLDivLoss(reduction='sum')
        loss = focal_loss
        #loss=ckld_loss
        #loss=nn.NLLLoss()
        epochs=300
        
        model = Net(slope)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        model=train(model.cuda(),loss,optimizer,num_epochs=epochs)
    
    
    
        tail=str(slope[0])+'_'+str(slope[1])+'_'+str(slope[2])+'____'+str(i)
        
        torch.save(model.state_dict(), dest_path_new+'model_slope'+tail+'.pth')
        
        temp=[]
        for idd in test_acc:
            temp.append(idd.cpu().detach().numpy())
        test_acc_new=np.array(temp)
        
        temp=[]
        for idd in train_acc:
            temp.append(idd.cpu().detach().numpy())
        train_acc_new=np.array(temp)
        
        
        
        plot()
    
        folders=[str(idd) for idd in range(num_classes)]#'0','1','2','3','4','5','6','7','8','9']
        evaluation_matrix=np.zeros([num_classes,num_classes])
        
        
        y_test_eval, y_pred= get_prediction(model)
        
        for idd in range(len(y_test_eval)):
             temp=int(y_test_eval[idd])
             evaluation_matrix[temp][y_pred[idd]]+=1
        df=pd.DataFrame(evaluation_matrix)
        df.columns=folders
        df.insert(0,'Classes',folders)
        df.to_csv(dest_path_new+'Confusion_matrix_'+tail+'.csv')
        
        
        
        evaluation_list=['Classes','TP','FP','TN','FN','Precision','Recall','Specificity','Accuracy','F1-score']
        #precision_recall={'TP':[0 for _ in range(len(folders))],'FP':[0 for _ in range(len(folders))],'TN':[0 for _ in range(len(folders))],'FN':[0 for _ in range(9)],'Precision':[0 for _ in range(len(folders))],'Recall':[0 for _ in range(folders)]}
        precision_recall={x:[0 for _ in range(num_classes)] for x in evaluation_list}
        precision_recall['Classes']=folders
        df_cm=pd.read_csv(dest_path_new+'Confusion_matrix_'+tail+'.csv')
        evaluation_matrix=df_cm[folders].values
        print("Test samples: ",np.sum(evaluation_matrix,axis=1))
        df_cm = pd.DataFrame(evaluation_matrix,index=folders,columns=folders)
        plt.figure(figsize=(50,50))
        
        #svm=sn.heatmap(df_cm,annot=True,cmap="Blues",robust=True)
        sn.set(font_scale=1.6) # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 24 }) # font size
        plt.savefig(dest_path_new+'conf_mat_'+tail+'.png', dpi=100)    
        #plt.show()
        
        for idd in range(num_classes):
            TN_temp = np.delete(evaluation_matrix, idd, axis=1)
            TN_temp = np.delete(TN_temp, idd, axis=0)
            precision_recall['TN'][idd]=np.sum(TN_temp)    
            precision_recall['TP'][idd]=evaluation_matrix[idd][idd]
            precision_recall['FP'][idd]=np.sum(evaluation_matrix[:,idd])-evaluation_matrix[idd][idd]
            precision_recall['FN'][idd]=np.sum(evaluation_matrix[idd,:])-evaluation_matrix[idd][idd]
            precision_recall['Precision'][idd]=precision_recall['TP'][idd]/np.sum(evaluation_matrix[:,idd])
            precision_recall['Recall'][idd]=precision_recall['TP'][idd]/np.sum(evaluation_matrix[idd,:])
            precision_recall['Specificity'][idd]=precision_recall['TN'][idd]/(precision_recall['TN'][idd]+precision_recall['FP'][idd])
                    
            precision_recall['Accuracy'][idd]=(precision_recall['TP'][idd]+precision_recall['TN'][idd])/np.sum(evaluation_matrix)*100
            precision_recall['F1-score'][idd]=200*precision_recall['Precision'][idd]*precision_recall['Recall'][idd]/(precision_recall['Precision'][idd]+precision_recall['Recall'][idd])
        
    
        dd=pd.DataFrame(precision_recall)
        dd.loc['mean']=dd.mean()
    #    dd['Mean F1-score']=np.mean(precision_recall['F1-score'])
        dd.to_csv(dest_path_new+'Results_'+tail+'.csv')
        dd=pd.read_csv(dest_path_new+'Results_'+tail+'.csv')
        dd=dd.rename(columns={'Unnamed: 0':'Number'})