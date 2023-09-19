#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision import transforms
import copy
from sklearn.model_selection import train_test_split


# In[2]:


margin = 1


# # Загружаем данные

# In[3]:


buy_patterns = np.load('data/buy_patterns.npy')


# In[4]:


buy_patterns_clusters = np.load('data/buy_patterns_clusters.npy')


# In[5]:


sell_patterns = np.load('data/sell_patterns.npy')


# In[6]:


sell_patterns_clusters = np.load('data/sell_patterns_clusters.npy')


# In[7]:


shift_sell_patterns_clusters = sell_patterns_clusters + 2


# # Объединяем датасеты

# In[8]:


add_num_data = 17


# In[9]:


patterns = np.concatenate((buy_patterns, sell_patterns), axis=0)
patterns = np.float32(patterns)


# In[11]:


patterns_clusters = np.concatenate((buy_patterns_clusters, shift_sell_patterns_clusters), axis=0)
patterns_clusters.shape


# In[76]:


def get_data(patterns_init, patterns_clusters_init, need_transform = True, need_reshape = True, split = 0.9, random_state = 42):
    
    patterns = copy.deepcopy(patterns_init)
    patterns_clusters = copy.deepcopy(patterns_clusters_init)
    
    if need_reshape:
        patterns = patterns.reshape(patterns.shape[0],1,patterns.shape[1],patterns.shape[2])
        
    X_train, X_test, y_train, y_test = train_test_split(patterns, patterns_clusters, test_size=(1-split), random_state=random_state)
    
    X_train = torch.tensor(X_train)
    X_test = torch.tensor(X_test)

    if need_transform:
        # #Изменяем размерность
        transform = transforms.Resize((28,28))
        X_train = transform(X_train)
        X_test = transform(X_test)
    
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)
    
    
    train_dataset = TensorDataset(X_train,y_train) # create your datset
    test_dataset = TensorDataset(X_test,y_test) # create your datset
    
    train_dataset.train = True
    train_dataset.transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.ToTensor()
    ])
    train_dataset.train_data = train_dataset.tensors[0]
    train_dataset.train_labels = train_dataset.tensors[1]
    train_dataset.targets = train_dataset.train_labels
    train_dataset.class_to_idx = {'0 - zero': 0,
     '1 - one': 1,
     '2 - two': 2,
     '3 - three': 3}
    train_dataset.classes = ['0 - zero', '1 - one','2 - two','3 - three']
    
    test_dataset.train = False
    test_dataset.transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.ToTensor()
    ])
    test_dataset.test_data = test_dataset.tensors[0]
    test_dataset.test_labels = test_dataset.tensors[1]
    test_dataset.targets = test_dataset.test_labels
    test_dataset.class_to_idx = {'0 - zero': 0,
     '1 - one': 1,
     '2 - two': 2,
     '3 - three': 3}
    train_dataset.classes = ['0 - zero', '1 - one','2 - two','3 - three']
    
    return train_dataset, test_dataset


# In[26]:


def get_data_v2(patterns_init, patterns_clusters_init, need_transform = True, need_reshape = True, split = 0.9):
    #Неверно разбивает лейблы датасета
    
    patterns = copy.deepcopy(patterns_init)
    patterns_clusters = copy.deepcopy(patterns_clusters_init)
    
    if need_reshape:
        patterns = patterns.reshape(patterns.shape[0],1,patterns.shape[1],patterns.shape[2])
    
    patterns = torch.tensor(patterns)

    if need_transform:
        # #Изменяем размерность
        transform = transforms.Resize((28,28))
        patterns = transform(patterns)
    
    print(patterns_clusters.shape)
    
    patterns_clusters = torch.tensor(patterns_clusters)
    
    
    dataset = TensorDataset(patterns,patterns_clusters) # create your datset
    
    train_size = int(split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_dataset.train = True
    train_dataset.transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.ToTensor()
    ])
    train_dataset.train_data = train_dataset.dataset.tensors[0]
    train_dataset.train_labels = train_dataset.dataset.tensors[1]
    train_dataset.targets = train_dataset.train_labels
    train_dataset.class_to_idx = {'0 - zero': 0,
     '1 - one': 1,
     '2 - two': 2,
     '3 - three': 3}
    train_dataset.classes = ['0 - zero', '1 - one','2 - two','3 - three']
    
    test_dataset.train = False
    test_dataset.transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.ToTensor()
    ])
    test_dataset.test_data = test_dataset.dataset.tensors[0]
    test_dataset.test_labels = test_dataset.dataset.tensors[1]
    print(len(test_dataset.test_labels))
    test_dataset.targets = test_dataset.test_labels
    test_dataset.class_to_idx = {'0 - zero': 0,
     '1 - one': 1,
     '2 - two': 2,
     '3 - three': 3}
    train_dataset.classes = ['0 - zero', '1 - one','2 - two','3 - three']
    
    return train_dataset, test_dataset


# In[27]:


train_dataset, test_dataset = get_data(patterns, patterns_clusters, need_transform = True, need_reshape = True, split = 0.9)


# In[28]:


n_classes = 4


# In[29]:


import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable

from trainer import fit
import numpy as np
cuda = torch.cuda.is_available()

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt

mnist_classes = ['0', '1', '2', '3']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

def plot_embeddings(embeddings, targets, xlim=None, ylim=None, pic_name = None):
    plt.figure(figsize=(10,10))
    for i in range(4):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(mnist_classes)
    
    if pic_name:
        plt.savefig('pic_result/'+pic_name+'.png')

def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels


# In[ ]:





# # Baseline: Classification with softmax

# In[30]:


# Set up data loaders
batch_size = 256
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# Set up the network and training parameters
from networks import EmbeddingNet, ClassificationNet
from metrics import AccumulatedAccuracyMetric

embedding_net = EmbeddingNet()
model = ClassificationNet(embedding_net, n_classes=n_classes)
if cuda:
    model.cuda()
loss_fn = torch.nn.NLLLoss()
lr = 1e-2
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 50


# In[31]:


fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[AccumulatedAccuracyMetric()])


# In[32]:


train_embeddings_baseline, train_labels_baseline = extract_embeddings(train_loader, model)
plot_embeddings(train_embeddings_baseline, train_labels_baseline, pic_name = 'baseline_train')
val_embeddings_baseline, val_labels_baseline = extract_embeddings(test_loader, model)
plot_embeddings(val_embeddings_baseline, val_labels_baseline, pic_name = 'baseline_val')


# In[ ]:





# # НОВЫЕ ДАТАСЕТЫ ДЛЯ НЕЙРОННЫХ СЕТЕЙ НИЖЕ

# In[33]:


train_dataset, test_dataset = get_data(patterns, patterns_clusters, need_transform = True, need_reshape = False, split = 0.9)


# # Siamese network

# In[34]:


# Set up data loaders
from datasets import SiameseMNIST

siamese_train_dataset = SiameseMNIST(train_dataset) # Returns pairs of images and target same/different
siamese_test_dataset = SiameseMNIST(test_dataset)
batch_size = 128
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
siamese_train_loader = torch.utils.data.DataLoader(siamese_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
siamese_test_loader = torch.utils.data.DataLoader(siamese_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# Set up the network and training parameters
from networks import EmbeddingNet, SiameseNet
from losses import ContrastiveLoss

margin = 1.
embedding_net = EmbeddingNet()
model = SiameseNet(embedding_net)
if cuda:
    model.cuda()
loss_fn = ContrastiveLoss(margin)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 100


# In[35]:


fit(siamese_train_loader, siamese_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)


# In[36]:


train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, model)
plot_embeddings(train_embeddings_cl, train_labels_cl, pic_name = 'siamese_train')
val_embeddings_cl, val_labels_cl = extract_embeddings(test_loader, model)
plot_embeddings(val_embeddings_cl, val_labels_cl, pic_name = 'siamese_val')


# # Triplet network

# In[37]:


# Set up data loaders
from datasets import TripletMNIST

triplet_train_dataset = TripletMNIST(train_dataset) # Returns triplets of images
triplet_test_dataset = TripletMNIST(test_dataset)
batch_size = 128
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# Set up the network and training parameters
from networks import EmbeddingNet, TripletNet
from losses import TripletLoss

margin = 1.
embedding_net = EmbeddingNet()
model = TripletNet(embedding_net)
if cuda:
    model.cuda()
loss_fn = TripletLoss(margin)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 100


# In[38]:


fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)


# In[39]:


train_embeddings_tl, train_labels_tl = extract_embeddings(train_loader, model)
plot_embeddings(train_embeddings_tl, train_labels_tl, pic_name = 'triplet_train')
val_embeddings_tl, val_labels_tl = extract_embeddings(test_loader, model)
plot_embeddings(val_embeddings_tl, val_labels_tl, pic_name = 'triplet_val')


# # ДАТАСЕТЫ ДЛЯ НЕЙРОННЫХ СЕТЕЙ НИЖЕ

# In[137]:


patterns_1 = np.concatenate((patterns, patterns), axis=0)

for i in range(100):
    patterns_1 = np.concatenate((patterns_1, patterns), axis=0)


# In[138]:


patterns_clusters_1 = np.concatenate((patterns_clusters, patterns_clusters), axis=0)

for i in range(100):
    patterns_clusters_1 = np.concatenate((patterns_clusters_1, patterns_clusters), axis=0)


# In[144]:


train_dataset, test_dataset = get_data(patterns_1, patterns_clusters_1, need_transform = True, need_reshape = True, split = 0.9, random_state = 50)


# In[145]:


# dir(train_dataset.train_data)


# In[146]:


# dir(train_dataset.__len__)


# # Online pair selection

# In[147]:


from datasets import BalancedBatchSampler

# We'll create mini batches by sampling labels that will be present in the mini batch and number of examples from each class
train_batch_sampler = BalancedBatchSampler(train_dataset.train_labels, n_classes=4, n_samples=2)
test_batch_sampler = BalancedBatchSampler(test_dataset.test_labels, n_classes=4, n_samples=2)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

# Set up the network and training parameters
from networks import EmbeddingNet
from losses import OnlineContrastiveLoss
from utils import AllPositivePairSelector, HardNegativePairSelector # Strategies for selecting pairs within a minibatch

margin = 1.
embedding_net = EmbeddingNet()
model = embedding_net
if cuda:
    model.cuda()
loss_fn = OnlineContrastiveLoss(margin, HardNegativePairSelector())
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 50


# In[148]:


fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)


# In[149]:


train_embeddings_ocl, train_labels_ocl = extract_embeddings(train_loader, model)
plot_embeddings(train_embeddings_ocl, train_labels_ocl, pic_name = 'one_pair_train')
val_embeddings_ocl, val_labels_ocl = extract_embeddings(test_loader, model)
plot_embeddings(val_embeddings_ocl, val_labels_ocl, pic_name = 'one_pair_val')


# # Online triplet selection

# In[133]:


from datasets import BalancedBatchSampler

# We'll create mini batches by sampling labels that will be present in the mini batch and number of examples from each class
train_batch_sampler = BalancedBatchSampler(train_dataset.train_labels, n_classes=4, n_samples=5)
test_batch_sampler = BalancedBatchSampler(test_dataset.test_labels, n_classes=4, n_samples=5)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

# Set up the network and training parameters
from networks import EmbeddingNet
from losses import OnlineTripletLoss
from utils import AllTripletSelector,HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector # Strategies for selecting triplets within a minibatch
from metrics import AverageNonzeroTripletsMetric

margin = 1.
embedding_net = EmbeddingNet()
model = embedding_net
if cuda:
    model.cuda()
loss_fn = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 50


# In[134]:


fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[AverageNonzeroTripletsMetric()])


# In[135]:


train_embeddings_otl, train_labels_otl = extract_embeddings(train_loader, model)
plot_embeddings(train_embeddings_otl, train_labels_otl, pic_name = 'online_triplet_train')
val_embeddings_otl, val_labels_otl = extract_embeddings(test_loader, model)
plot_embeddings(val_embeddings_otl, val_labels_otl, pic_name = 'online_triplet_train')


# In[136]:


x_lim = (np.min(train_embeddings_ocl[:,0]), np.max(train_embeddings_ocl[:,0]))
y_lim = (np.min(train_embeddings_ocl[:,1]), np.max(train_embeddings_ocl[:,1]))
plot_embeddings(train_embeddings_cl, train_labels_cl, x_lim, y_lim)
plot_embeddings(train_embeddings_ocl, train_labels_ocl, x_lim, y_lim)


# In[ ]:





# In[ ]:





# In[ ]:




