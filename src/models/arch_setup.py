from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from src.visualization import metrics
from src.features import specto_feat


class DS(Dataset):
    def __init__(self,df):
        """
        Arguments:
        df -- {dataframe} -- data. expected columns: target_type (labels), doppler_burst, iq_sweep_burst, augmentation_info

        index is expected to be in ascending order, but it might contain holes!
        index should be the same as segment_id.
        this is important when locating the segment of the augmentations.
        """

        super().__init__()
        self.df=df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        data_inner = self.df.iloc[idx].copy()  # use iloc here because must get absolute row position

        if data_inner.iq_sweep_burst is None:

            iq_matrix = None
            doppler_vector = None

            for augment_info in data_inner.augmentation_info:

                #print(f"augment_info:{augment_info}")

                if augment_info['type']=='shift':

                    #print(f"shift")

                    iq_list = []
                    dopller_list = []
                    from_segments = augment_info['from_segments']
                    shift_by = augment_info['shift']

                    for i in from_segments:
                        iq_list.append(self.df.loc[i]['iq_sweep_burst'])     # use loc here because we need the actual segment id (by index)
                        dopller_list.append(self.df.loc[i]['doppler_burst'])

                    #print(f"iq_list:{iq_list},dopller_list:{dopller_list}. shape:{iq_list[0].shape}. len:{len(iq_list)}")

                    iq_matrix = np.concatenate(iq_list, axis=1)  # 2*(128,32) => (128,64)
                    doppler_vector = np.concatenate(dopller_list, axis=0)  # 2*(32,1) => (64,1)

                    # cut the iq_matrix according to the shift
                    iq_matrix = iq_matrix[:,shift_by:shift_by+32]
                    doppler_vector = doppler_vector[shift_by:shift_by+32]

                if iq_matrix is None and augment_info['type']=='flip':

                    #print(f"flip")

                    from_segment = augment_info['from_segment']

                    iq_matrix = self.df[from_segment].iq_sweep_burst
                    doppler_vector = self.df[from_segment].doppler_vector

                #print(f"iq_matrix:{iq_matrix},doppler_vector:{doppler_vector}")

            data_inner.iq_sweep_burst = iq_matrix
            data_inner.doppler_burst = doppler_vector


        #print(f"data_inner:{data_inner}")

        # convert to structure supported by preprocess method
        data_inner_o = {k:[v] for (k,v) in data_inner.to_dict().items()}

        # do preprocess
        data = specto_feat.data_preprocess(data_inner_o)

        # augementations
        # do flips (if needed)
        for augment_info in data['augmentation_info'][0]: # the [0] is because we added [] in the data_inner_o
            #print(f"augment_info:{augment_info}")
            if augment_info['type']=='flip':
                if augment_info['mode']=='veritcal':
                    data['iq_sweep_burst'] = np.flip(data_inner.iq_sweep_burst,0)
                    data['doppler_burst'] = np.abs(128-data_inner.doppler_burst)

                if augment_info['mode']=='horizontal':
                    data['iq_sweep_burst'] = np.flip(data_inner.iq_sweep_burst,1)
                    data['doppler_burst'] = np.flip(data_inner.doppler_burst,1)

        label2model = 0 if data['target_type']=='animal' else 1
        data2model = data['iq_sweep_burst']

        #print(f"data2model:{data2model.shape}")  # (1,132,28)
        #data2model = data2model.reshape(list(data2model.shape)+[1])
        data2model = np.expand_dims(data2model.squeeze(),axis=2)  # (132,28,1)

        return data2model, label2model


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
