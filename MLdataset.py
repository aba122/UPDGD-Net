from torch.utils.data import Dataset, DataLoader
import scipy.io
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize, scale
import math,random
from sklearn.utils import shuffle


def loadMvMlDataFromMat(mat_path):
    data = scipy.io.loadmat(mat_path)
    mv_data = data['X'][0]
    labels = data['label']
    labels = labels.astype(np.float32)
    if labels.min() == -1:
        labels = (labels + 1) * 0.5
    if labels.shape[0] in mv_data[0].shape:
        total_sample_num = labels.shape[0]
    elif labels.shape[1] in mv_data[0].shape:
        total_sample_num = labels.shape[0]
    if total_sample_num!=mv_data[0].shape[0]:
        mv_data = [v_data.T for v_data in mv_data]
    if total_sample_num!=labels.shape[0]:
        labels = labels.T
    assert mv_data[0].shape[0]==labels.shape[0]==total_sample_num
    ind00 = labels==0
    # mv_data = [np.delete(v_data, ind00,axis=0)) for v_data in mv_data]
    # labels = np.delete(labels,ind00,axis=0))

    mv_data = [StandardScaler().fit_transform(v_data.astype(np.float32)) for v_data in mv_data]
    # shuffle the data list
    random.seed(1)
    rand_index=list(range(total_sample_num))
    random.shuffle(rand_index)
    return [v_data[rand_index] for v_data in mv_data],labels[rand_index],total_sample_num

def loadMfDIMvMlDataFromMat(mat_path, fold_mat_path,fold_idx=0):
    # load multiple folds double incomplete multi-view multi-label data and labels 
    # mark sure the out dimension is n x d, where n is the number of samples
    data = scipy.io.loadmat(mat_path)
    datafold = scipy.io.loadmat(fold_mat_path)
    # multi-view data labels
    mv_data = data['X'][0]
    labels = data['label']
    labels = labels.astype(np.float32)
    if labels.min() == -1:
            labels = (labels + 1) * 0.5
    if labels.shape[0] in mv_data[0].shape:
        total_sample_num = labels.shape[0]
    elif labels.shape[1] in mv_data[0].shape:
        total_sample_num = labels.shape[0]
    if total_sample_num!=mv_data[0].shape[0]:
        mv_data = [v_data.T for v_data in mv_data]
    if total_sample_num!=labels.shape[0]:
        labels = labels.T
    assert mv_data[0].shape[0]==labels.shape[0]==total_sample_num
    
    folds_data = datafold['folds_data']
    folds_label = datafold['folds_label']
    folds_sample_index = datafold['folds_sample_index']
    # incomplete data, label and random_sample index
    inc_view_indicator = np.array(folds_data[0, fold_idx], 'int32')
    #inc_view_indicator = np.ones_like(inc_view_indicator)
    inc_label_indicator = np.array(folds_label[0, fold_idx], 'int32') 
    #inc_label_indicator = np.ones_like(inc_label_indicator) # incomplete label index
    sample_index = np.array(folds_sample_index[0, fold_idx], 'int32').reshape(-1)-1 # index start from 0
    labels,inc_view_indicator,inc_label_indicator = labels[sample_index],inc_view_indicator[sample_index],inc_label_indicator[sample_index]
    mv_data = [v_data[sample_index,:] for v,v_data in enumerate(mv_data)]

    assert inc_view_indicator.shape[0]==inc_label_indicator.shape[0]==sample_index.shape[0]==labels.shape[0]
    
    #Shuffle the data and labels together in the same random order
   # mv_data, labels, inc_view_indicator, inc_label_indicator = shuffle(mv_data,labels,inc_view_indicator,inc_label_indicator,random_state=42)
    
    # incomplete data construction and normalization
    nor_mv_data = [(StandardScaler().fit_transform(v_data.astype(np.float32))) for v,v_data in enumerate(mv_data)]
    inc_mv_data = [np.random.randn(v_data.shape[0],v_data.shape[1]) for v_data in nor_mv_data]
    for v,v_data in enumerate(nor_mv_data):
        inc_mv_data[v][inc_view_indicator[:,v]==1,:] = v_data[inc_view_indicator[:,v]==1,:].copy()
    # incomplete label construction
    inc_labels = labels*inc_label_indicator
    # delete data with all zero label 
    ind00 = labels.sum(axis=1)==0


    return inc_mv_data,inc_labels,labels,inc_view_indicator,inc_label_indicator,total_sample_num
    
class ComDataset(Dataset):
    def __init__(self,mat_path,training_ratio=0.7,val_ratio=0.15,mode='train',semisup=False):
        self.mv_data, self.labels, self.total_sample_num= loadMvMlDataFromMat(mat_path)
        self.train_sample_num = math.ceil(self.total_sample_num * training_ratio)
        self.val_sample_num = math.ceil(self.total_sample_num * val_ratio)
        self.test_sample_num = self.total_sample_num - self.train_sample_num - self.val_sample_num
        if mode=='train':
            self.cur_mv_data = [v_data[:self.train_sample_num] for v_data in self.mv_data]
            self.cur_labels = self.labels[:self.train_sample_num]
        elif mode=='val':
            self.cur_mv_data = [v_data[self.train_sample_num:self.train_sample_num+self.val_sample_num] for v_data in self.mv_data]
            self.cur_labels = self.labels[self.train_sample_num:self.train_sample_num+self.val_sample_num]
        else:
            self.cur_mv_data = [v_data[self.train_sample_num+self.val_sample_num:] for v_data in self.mv_data]
            self.cur_labels = self.labels[self.train_sample_num+self.val_sample_num:]
        # print('is_train:',is_train,'num:',self.cur_mv_data[0].shape)
        self.mode = mode
        self.classes_num = self.labels.shape[1]
        self.d_list = [da.shape[1] for da in self.mv_data]
    def __len__(self):
        if self.mode == 'train':
            return self.train_sample_num 
        elif self.mode == 'val':
            return self.val_sample_num 
        else: return self.test_sample_num 
    
    def __getitem__(self, index):
        # index = index if self.is_train else self.train_sample_num+index
        data = [torch.tensor(v[index],dtype=torch.float) for v in self.cur_mv_data] 
        label = torch.tensor(self.cur_labels[index], dtype=torch.float)
        return data,label, data, label

class IncDataset(Dataset):
    def __init__(self,mat_path, fold_mat_path, training_ratio=0.7, val_ratio=0.15, fold_idx=0, mode='train',semisup=False):
        inc_mv_data, inc_labels, labels, inc_V_ind, inc_L_ind, total_sample_num= loadMfDIMvMlDataFromMat(mat_path,fold_mat_path,fold_idx)
        #inc_labels, labels, inc_V_ind, inc_L_ind = shuffle(inc_labels, labels, inc_V_ind, inc_L_ind, random_state=42)
        # inc_mv_data, inc_labels, labels, inc_V_ind, inc_L_ind, total_sample_num= loadMvMlDataFromMat(mat_path)
        self.train_sample_num = math.ceil(total_sample_num * training_ratio)
        self.val_sample_num = math.ceil(total_sample_num * val_ratio)
        self.test_sample_num = total_sample_num - self.train_sample_num - self.val_sample_num

        if mode=='train':
            self.cur_mv_data = [v_data[:self.train_sample_num] for v_data in inc_mv_data]
            self.cur_labels = inc_labels[:self.train_sample_num]
            self.cur_inc_V_ind = inc_V_ind[:self.train_sample_num]
            self.cur_inc_L_ind = inc_L_ind[:self.train_sample_num]
            self.indices = np.arange(self.train_sample_num)
            
        elif mode=='val':
            self.cur_mv_data = [v_data[self.train_sample_num:self.train_sample_num+self.val_sample_num] for v_data in inc_mv_data]
            self.cur_labels = labels[self.train_sample_num:self.train_sample_num+self.val_sample_num]
            self.cur_inc_V_ind = inc_V_ind[self.train_sample_num:self.train_sample_num+self.val_sample_num]
            self.cur_inc_L_ind = np.ones_like(inc_L_ind[self.train_sample_num:self.train_sample_num+self.val_sample_num])
            self.indices = np.arange(self.train_sample_num, self.train_sample_num + self.val_sample_num)
        else:
            self.cur_mv_data = [v_data[self.train_sample_num+self.test_sample_num:] for v_data in inc_mv_data]
            self.cur_labels = labels[self.train_sample_num+self.test_sample_num:]
            self.cur_inc_V_ind = inc_V_ind[self.train_sample_num+self.test_sample_num:]
            self.cur_inc_L_ind = np.ones_like(inc_L_ind[self.train_sample_num+self.test_sample_num:])
            self.indices = np.arange(self.train_sample_num + self.test_sample_num, total_sample_num)


        self.mode = mode
        self.classes_num = labels.shape[1]
        self.d_list = [da.shape[1] for da in inc_mv_data]

    def __len__(self):
        if self.mode == 'train':
            return self.train_sample_num 
        elif self.mode == 'val':
            return self.val_sample_num 
        else: return self.test_sample_num 
    
    def __getitem__(self, index):
        # index = index if self.is_train else self.train_sample_num+index
        data = [torch.tensor(v[index],dtype=torch.float) for v in self.cur_mv_data] 
        label = torch.tensor(self.cur_labels[index], dtype=torch.float)
        inc_V_ind = torch.tensor(self.cur_inc_V_ind[index], dtype=torch.int32)
        inc_L_ind = torch.tensor(self.cur_inc_L_ind[index], dtype=torch.int32)
        orig_index = self.indices[index]
        return data,label,inc_V_ind,inc_L_ind,orig_index

def getComDataloader(matdata_path,training_ratio=0.7,val_ratio=0.15,mode='train',batch_size=1,num_workers=1,shuffle=False):
    dataset = ComDataset(matdata_path, training_ratio=training_ratio, val_ratio=val_ratio, mode=mode)
    dataloder = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
    return dataloder,dataset
 
def getIncDataloader(matdata_path, fold_matdata_path, training_ratio=0.7, val_ratio=0.15, fold_idx=0, mode='train',batch_size=1,num_workers=1,shuffle=True):
    dataset = IncDataset(matdata_path, fold_matdata_path, training_ratio=training_ratio, val_ratio=val_ratio, mode=mode, fold_idx=fold_idx)
    dataloder = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
    return dataloder,dataset
    
if __name__=='__main__':
    # dataloder,dataset = getComDataloader('/disk/MATLAB-NOUPLOAD/MyMVML-data/corel5k/corel5k_six_view.mat',training_ratio=0.7,mode='train',batch_size=3,num_workers=2)
    dataloder,dataset = getIncDataloader('/disk/MATLAB-NOUPLOAD/MyMVML-data/corel5k/corel5k_six_view.mat','/disk/MATLAB-NOUPLOAD/MyMVML-data/corel5k/corel5k_six_view_MaskRatios_0.5_LabelMaskRatio_0.5_TraindataRatio_0.7.mat',training_ratio=0.7,mode='train',batch_size=3,num_workers=2)
    print(dataset.test_sample_num)
    labels = torch.tensor(dataset.cur_labels).float()
    # total_graph = torch.matmul(labels.T,labels)
    # print(dataset.__getitem__(0)[1])
    # total_graph = torch.zeros([dataset.cur_labels.shape[1],dataset.cur_labels.shape[1]])
    # total_label = []
    # for i,(data,label,inc_V_ind,inc_L_ind) in enumerate(dataloder):
    #     label = label.float()
    #     total_label = torch.cat((total_label,label),dim=0) if len(total_label)>0 else label
    #     valid_labels_sum = torch.matmul(inc_L_ind.T.float(), inc_L_ind.float()) #[c, c]
    #     graph = torch.matmul(label.T, label) 
    #     # print(torch.matmul(label.T, label)[0,0],label_sum_cx1[0,0])
    #     total_graph += graph
    # total_graph = total_graph/(torch.diag(total_graph).unsqueeze(1))
    # np.save('c.npy',total_graph.fill_diagonal_(0).numpy())

    # loadMfDIMvMlDataFromMat('/disk1/lcl/MATLAB-NOUPLOAD/MyMVML-data/corel5k/corel5k_six_view.mat','/disk1/lcl/MATLAB-NOUPLOAD/MyMVML-data/corel5k/corel5k_six_view_MaskRatios_0.5_LabelMaskRatio_0.5_TraindataRatio_0.7.mat',fold_idx=0)
