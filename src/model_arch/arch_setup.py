from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc



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
        value_s = value if type(value)=="int" else "{:.4f}".format(value)
        print(f"{key} : {value_s}, ",end="")
    print("\n---------------------------\n")

def thresh(output, thresh_hold = 0.5):
    return [0 if x <thresh_hold else 1 for x in output]


def accuracy_calc(outputs, labels):
    #print("acc1:",outputs, labels)
    preds = thresh(outputs)
    #print("acc2:",preds)
    return np.sum(preds == labels) / len(preds)

def train_epochs(tr_loader,val_loader,model,criterion,optimizer, num_epochs, device,train_y,val_y):

    training_log =[]

    for epoch in range(num_epochs):

        print("started training epoch no. {}".format(epoch+1))

        tr_loss = 0
        tr_size = 0
        tr_y_hat = np.array([])
        tr_labels = np.array([])

        #train loop
        for step,batch in enumerate(tr_loader):

            data, labels = batch
            tr_labels = np.append(tr_labels,labels)

            data = data.to(device,dtype=torch.float32)
            labels = labels.to(device,dtype=torch.float32)
            outputs = model(data)

            labels = labels.view(-1,1)
            outputs = outputs.view(-1,1)

            loss = criterion(outputs,labels)
            loss.backward()

            tr_loss+=loss.item()
            tr_size+=data.shape[0]

            tr_y_hat = np.append(tr_y_hat,outputs.detach().cpu().numpy())

            optimizer.step()
            optimizer.zero_grad()

        val_loss = 0
        val_size = 0
        val_y_hat = np.array([])
        val_labels = np.array([])

        #validation loop
        for step, batch in enumerate(val_loader):

            data, labels = batch
            val_labels = np.append(val_labels,labels)

            data = data.to(device,dtype=torch.float32)
            labels = labels.to(device,dtype=torch.float32)
            outputs = model(data)

            labels = labels.view(-1,1)
            outputs = outputs.view(-1,1)

            loss = criterion(outputs,labels)

            val_loss += loss.item()
            val_size += data.shape[0]

            val_y_hat = np.append(val_y_hat,outputs.detach().cpu().numpy())


        tr_fpr, tr_tpr, _ = roc_curve(tr_labels, tr_y_hat)
        val_fpr, val_tpr, _ = roc_curve(val_labels, val_y_hat)

        epoch_log = {'epoch': epoch,
                     'loss': tr_loss ,
                     'auc': auc(tr_fpr, tr_tpr),
                     'acc': accuracy_calc(tr_y_hat,tr_labels),
                     'val_loss': val_loss ,
                     'val_auc': auc(val_fpr,val_tpr),
                     'val_acc': accuracy_calc(val_y_hat,val_labels)}


        pretty_log(epoch_log)

        training_log.append(epoch_log)

    #return training_log   


def plot_loss_train_test(trainloss,test_loss):
    plt.hist(trainloss)
    plt.hist(test_loss)
    plt.tight_lat