from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score



class DS(Dataset):
    def __init__(self,df,labels):
        super().__init__()
        self.df=df
        self.labels=labels
 
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        data = self.df[idx]
        label = self.labels[idx]        
        return data,label

def pretty_log(log):
    for key,value in log.items():
        print("{} : {:.4f}".format(key,value))
    print("\n---------------------------\n")

def thresh(output, thresh_hold = 0.5):
    return [0 if x <thresh_hold else 1 for x in output]

def train_epochs(tr_loader,val_loader,model,criterion,optimizer, num_epochs, train_y,val_y):

    using_cuda = False

    if using_cuda:    
        if torch.has_cuda:
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu:0')
    else:
        device = torch.device('cpu:0')

    model.to(device)

    training_log =[]

    for epoch in range(num_epochs):

        print("started training epoch no. {}".format(epoch+1))

        tr_loss = 0
        tr_size = 0
        tr_y_hat = np.array([])

        for step,batch in enumerate(tr_loader):

            data, labels = batch

            data = data.to(device,dtype=torch.float32)
            labels = labels.to(device,dtype=torch.float32)
            outputs = model(data)
        
            loss = criterion(outputs,labels)
            loss.backward()
            
            tr_loss+=loss.item()
            tr_size+=data.shape[0]

            tr_y_hat = np.append(tr_y_hat,thresh(outputs.detach().numpy()))

            optimizer.step()
            optimizer.zero_grad()

        val_loss = 0
        val_size = 0
        val_y_hat = np.array([])

        for step, batch in enumerate(val_loader):
            
            data, labels = batch
            data = data.to(device,dtype=torch.float32)
            labels = labels.to(device,dtype=torch.float32)
            outputs = model(data)
            
            loss = criterion(outputs,labels)

            val_loss += loss.item()
            val_size += data.shape[0]

            val_y_hat = np.append(val_y_hat,thresh(outputs.detach().numpy()))


        tr_fpr, tr_tpr, _ = roc_curve(train_y, tr_y_hat)
        val_fpr, val_tpr, _ = roc_curve(val_y, val_y_hat)

        epoch_log = {'epoch': epoch,
                     'loss': tr_loss / tr_size,
                     'auc': auc(tr_fpr, tr_tpr),
                     'acc': accuracy_score(train_y,tr_y_hat),
                     'val_loss': val_loss / val_size,
                     'val_auc': auc(val_fpr,val_tpr),
                     'val_acc': accuracy_score(val_y,val_y_hat)}


        pretty_log(epoch_log)

        training_log.append(epoch_log)

    return training_log