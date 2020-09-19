from torch.utils.data import Dataset, IterableDataset, DataLoader
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.visualization import metrics


class DS(Dataset):
    def __init__(self, df, labels, config, addit=None):
        super().__init__()
        self.df = df
        self.labels = labels
        self.config = config
        if addit:
            self.addit = np.array(addit)
        else:
            self.addit = None

    def __len__(self):
    # TODO implement len of this dataset
        pass

    def __getitem__(self, idx):
    # TODO implement creation of shifts and flips in this dataset, which will be used for DS2 class __iter__ method
        pass


class DS2(IterableDataset):
    def __init__(self, df, labels, config):
        '''
             arguments:
             ...
             config -- {dict}:
                 num_tracks -- {int} -- # of tracks to take from aux dataset
                 valratio -- {int} -- Ratio of train/val split
                 get_shifts -- {bool} -- Flag to add shifts
                 shift_segment -- {int} -- How much to shift tracks to generate new segments
                 get_horizontal_flip -- {bool} -- Flag to add horizontal flips
                 get_vertical_flip -- {bool} -- Flag to add vertical flips
                 block_size -- {int} -- Max number of samples allowed to be held in a memory 
        '''
        super().__init__()
        self.df = df
        self.labels = labels
        self.config = config


    def __iter__(self):
        # TODO implement function that takes a track and turns it into it's own DS dataset
        assert False



def pretty_log(log):
    for key,value in log.items():
        value_s = value if type(value)=="int" else "{:.4f}".format(value)
        print("{} : {}, ".format(key, value_s),end="")
    print("\n---------------------------\n")

def thresh(output, thresh_hold = 0.5):
    return [0 if x <thresh_hold else 1 for x in output]


def accuracy_calc(outputs, labels):
    #print("acc1:",outputs, labels)
    preds = thresh(outputs)
    #print("acc2:",preds)
    return np.sum(preds == labels) / len(preds)

def train_epochs(tr_loader,val_loader,model,criterion,optimizer, num_epochs, device,train_y,val_y,log=None,WANDB_enable = False,wandb=None):

    # If we want to run more epochs, want to keep the same log of the old model
    if log:
        training_log = log
    else:
        training_log = []

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
            snr = None  #added

            #added
            if isinstance(data, list):
              snr = data[1].to(device,dtype=torch.float32)
              data = data[0]

            data = data.to(device,dtype=torch.float32)
            labels = labels.to(device,dtype=torch.float32)
            
            # added
            if snr:
              outputs = model(data,snr)
            else:
              outputs = model(data)

            labels = labels.view(-1,1)
            outputs = outputs.view(-1,1)

            loss = criterion(outputs,labels)
            loss.backward()


            tr_loss+=loss.item()
            tr_size+=data.shape[0]

            if torch.cuda.is_available():
                tr_y_hat = np.append(tr_y_hat,outputs.detach().cpu().numpy())
            else:
                tr_y_hat = np.append(tr_y_hat,outputs.detach().numpy())

            optimizer.step()
            optimizer.zero_grad()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
            if isinstance(data, list):
              snr = data[1].to(device,dtype=torch.float32)
              data = data[0]
            data = data.to(device,dtype=torch.float32)
            labels = labels.to(device,dtype=torch.float32)
            if snr is not None:
              outputs = model(data,snr)
            else:
              outputs = model(data)
            labels = labels.view(-1,1)
            outputs = outputs.view(-1,1)

            loss = criterion(outputs,labels)

            val_loss += loss.item()
            val_size += data.shape[0]

            if torch.cuda.is_available():
                val_y_hat = np.append(val_y_hat,outputs.detach().cpu().numpy())
            else:
                val_y_hat = np.append(val_y_hat,outputs.detach().numpy())

        tr_fpr, tr_tpr, _ = roc_curve(tr_labels, tr_y_hat)
        val_fpr, val_tpr, _ = roc_curve(val_labels, val_y_hat)

        epoch_log = {'epoch': epoch+1,
                     'loss': tr_loss ,
                     'auc': auc(tr_fpr, tr_tpr),
                     'acc': accuracy_calc(tr_y_hat,tr_labels),
                     'val_loss': val_loss ,
                     'val_auc': auc(val_fpr,val_tpr),
                     'val_acc': accuracy_calc(val_y_hat,val_labels)}


        pretty_log(epoch_log)

        training_log.append(epoch_log)

        if WANDB_enable == True:
            wandb.log(epoch_log)

    return training_log   


def plot_loss_train_test(logs,model):
    tr_loss = []
    val_loss = []
    for epoch_log in logs:
        tr_loss.append(epoch_log['loss'])
        val_loss.append(epoch_log['val_loss'])

    plt.figure(figsize=(12,8))
    plt.title(model._get_name())
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(range(len(tr_loss)),tr_loss,label="Train");
    plt.plot(range(len(val_loss)),val_loss,label="Val");
    plt.legend()
    plt.tight_layout()
    plt.show();


def plot_ROC_local_gpu(train_loader, val_loader, model,device):
    '''
    Working on a local GPU, there is limited space and therefore a need to run the ROC examples in batches.

    Outputs ROC plot as defined in utils.stats

    Arguments:
        train_loader -- {DataLoader} -- has train data stored in batches defined in notebook
        val_loader -- {DataLoader} -- has val data stored in batches defined in notebook
        model -- {nn.Module} -- pytorch model 
        device -- {torch.device} -- cpu/cuda

    '''
    tr_y = np.array([])
    tr_y_hat = np.array([])
    vl_y = np.array([])
    vl_y_hat = np.array([])

    for data,label in train_loader:
        tr_y_hat = np.append(tr_y_hat,np.array(thresh(model(data.to(device).type(torch.float32)).detach().cpu())))
        tr_y = np.append(tr_y, np.array(label.detach().cpu()))

    for data,label in val_loader:
        vl_y_hat = np.append(vl_y_hat, np.array(thresh(model(data.to(device).type(torch.float32)).detach().cpu())))
        vl_y = np.append(vl_y,np.array(label.detach().cpu()))


    pred = [tr_y_hat,vl_y_hat]
    actual = [tr_y,vl_y]
    metrics.stats(pred, actual)


def plot_ROC(train_x, val_x, train_y, val_y, model,device):
    '''
    Outputs ROC plot as defined in utils.stats

    Arguments:
        train_x -- {np.array} -- train data
        val_x -- {np.array} --  val data
        train_y -- {np.array} -- train labels
        val_y -- {np.array} -- val labels
        model -- {nn.Module} -- pytorch model 
        device -- {torch.device} -- cpu/cuda

    '''
    x1 = thresh(model(torch.from_numpy(train_x).to(device).type(torch.float32)).detach().cpu())
    x2 = thresh(model(torch.from_numpy(val_x).to(device).type(torch.float32)).detach().cpu())

    pred = [x1,x2]

    actual = [train_y, val_y]
    metrics.stats(pred, actual)
