import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from torch.utils.data import Dataset





def new_relu(y,slope=1):
    return F.relu(y)*slope


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
