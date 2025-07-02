#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Libraries

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd
import torch 
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import random


# In[2]:


# Check torch version
print('Torch Version:',torch.__version__)
print()

# Check torch cuda compatibility 
print('Torch Cuda Compatibility:',torch.version.cuda)
print()

# Check if GPU is available
print('Is the GPU available?', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
# print('Device name:', torch.cuda.get_device_name(0))
# print(torch.cuda.get_arch_list())


# In[3]:


# CHECK NUMBER OF GPUs

print(torch.cuda.device_count())


# In[4]:


# Device Object - CPU and GPU 
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device", device)


# In[5]:


# Load Data - YH16

# 03/06
YH16_1_X_L = pd.read_csv('Kinematics Data/YH16_240306_Kinematics_All.csv')
YH16_1_Y_L = pd.read_csv('Neural Data/YH16_240306_L_IRN_Single_Trial.csv')

YH16_1_X_R = pd.read_csv('Kinematics Data/YH16_240306_Kinematics_All.csv')
YH16_1_Y_R = pd.read_csv('Neural Data/YH16_240306_R_IRN_Single_Trial.csv')

# 03/07
YH16_2_X_L = pd.read_csv('Kinematics Data/YH16_240307_Kinematics_All.csv')
YH16_2_Y_L = pd.read_csv('Neural Data/YH16_240307_L_IRN_Single_Trial.csv')

YH16_2_X_R = pd.read_csv('Kinematics Data/YH16_240307_Kinematics_All.csv')
YH16_2_Y_R = pd.read_csv('Neural Data/YH16_240307_R_IRN_Single_Trial.csv')

# 03/08
YH16_3_X_L = pd.read_csv('Kinematics Data/YH16_240308_Kinematics_All.csv')
YH16_3_Y_L = pd.read_csv('Neural Data/YH16_240308_L_IRN_Single_Trial.csv')

YH16_3_X_R = pd.read_csv('Kinematics Data/YH16_240308_Kinematics_All.csv')
YH16_3_Y_R = pd.read_csv('Neural Data/YH16_240308_R_IRN_Single_Trial.csv')


# In[6]:


# Load Data - YH28

# 06/28
YH28_1_X_L = pd.read_csv('Kinematics Data/YH28_240628_Kinematics_All.csv')
YH28_1_Y_L = pd.read_csv('Neural Data/YH28_240628_L_IRN_Single_Trial_QC.csv')

YH28_1_X_R = pd.read_csv('Kinematics Data/YH28_240628_Kinematics_All.csv')
YH28_1_Y_R = pd.read_csv('Neural Data/YH28_240628_R_IRN_Single_Trial_QC.csv')

# 06/30
YH28_2_X_L = pd.read_csv('Kinematics Data/YH28_240630_Kinematics_All.csv')
YH28_2_Y_L = pd.read_csv('Neural Data/YH28_240630_L_IRN_Single_Trial_QC.csv')

YH28_2_X_R = pd.read_csv('Kinematics Data/YH28_240630_Kinematics_All.csv')
YH28_2_Y_R = pd.read_csv('Neural Data/YH28_240630_R_IRN_Single_Trial_QC.csv')

# 07/01
YH28_3_X_L = pd.read_csv('Kinematics Data/YH28_240701_Kinematics_All.csv')
YH28_3_Y_L = pd.read_csv('Neural Data/YH28_240701_L_IRN_Single_Trial_QC.csv')

YH28_3_X_R = pd.read_csv('Kinematics Data/YH28_240701_Kinematics_All.csv')
YH28_3_Y_R = pd.read_csv('Neural Data/YH28_240701_R_IRN_Single_Trial_QC.csv')

# 07/02
YH28_4_X_L = pd.read_csv('Kinematics Data/YH28_240702_Kinematics_All.csv')
YH28_4_Y_L = pd.read_csv('Neural Data/YH28_240702_L_IRN_Single_Trial_QC.csv')

YH28_4_X_R = pd.read_csv('Kinematics Data/YH28_240702_Kinematics_All.csv')
YH28_4_Y_R = pd.read_csv('Neural Data/YH28_240702_R_IRN_Single_Trial_QC.csv')


# In[7]:


# Load Data - YH31

# 08/03
YH31_1_X_L = pd.read_csv('Kinematics Data/YH31_240803_Kinematics_All.csv')
YH31_1_Y_L = pd.read_csv('Neural Data/YH31_240803_L_IRN_Single_Trial_QC.csv')

YH31_1_X_R = pd.read_csv('Kinematics Data/YH31_240803_Kinematics_All.csv')
YH31_1_Y_R = pd.read_csv('Neural Data/YH31_240803_R_IRN_Single_Trial_QC.csv')

# 08/04
YH31_2_X_L = pd.read_csv('Kinematics Data/YH31_240804_Kinematics_All.csv')
YH31_2_Y_L = pd.read_csv('Neural Data/YH31_240804_L_IRN_Single_Trial_QC.csv')

YH31_2_X_R = pd.read_csv('Kinematics Data/YH31_240804_Kinematics_All.csv')
YH31_2_Y_R = pd.read_csv('Neural Data/YH31_240804_R_IRN_Single_Trial_QC.csv')

# 08/05
YH31_3_X_L = pd.read_csv('Kinematics Data/YH31_240805_Kinematics_All.csv')
YH31_3_Y_L = pd.read_csv('Neural Data/YH31_240805_L_IRN_Single_Trial_QC.csv')

YH31_3_X_R = pd.read_csv('Kinematics Data/YH31_240805_Kinematics_All.csv')
YH31_3_Y_R = pd.read_csv('Neural Data/YH31_240805_R_IRN_Single_Trial_QC.csv')

# 08/06
YH31_4_X_L = pd.read_csv('Kinematics Data/YH31_240806_Kinematics_All.csv')
YH31_4_Y_L = pd.read_csv('Neural Data/YH31_240806_L_IRN_Single_Trial_QC.csv')

YH31_4_X_R = pd.read_csv('Kinematics Data/YH31_240806_Kinematics_All.csv')
YH31_4_Y_R = pd.read_csv('Neural Data/YH31_240806_R_IRN_Single_Trial_QC.csv')

# 08/07
YH31_5_X_L = pd.read_csv('Kinematics Data/YH31_240807_Kinematics_All.csv')
YH31_5_Y_L = pd.read_csv('Neural Data/YH31_240807_L_IRN_Single_Trial_QC.csv')

YH31_5_X_R = pd.read_csv('Kinematics Data/YH31_240807_Kinematics_All.csv')
YH31_5_Y_R = pd.read_csv('Neural Data/YH31_240807_R_IRN_Single_Trial_QC.csv')


# In[8]:


# Load Data - YH32

# 08/15
YH32_1_X_L = pd.read_csv('Kinematics Data/YH32_240815_Kinematics_All.csv')
YH32_1_Y_L = pd.read_csv('Neural Data/YH32_240815_L_IRN_Single_Trial_QC.csv')

# 08/22
YH32_2_X_R = pd.read_csv('Kinematics Data/YH32_240822_Kinematics_All.csv')
YH32_2_Y_R = pd.read_csv('Neural Data/YH32_240822_R_IRN_Single_Trial_QC.csv')

# 08/23
YH32_3_X_R = pd.read_csv('Kinematics Data/YH32_240823_Kinematics_All.csv')
YH32_3_Y_R = pd.read_csv('Neural Data/YH32_240823_R_IRN_Single_Trial_QC.csv')


# In[9]:


# Load Data - YH38

# 09/28
YH38_1_X_L = pd.read_csv('Kinematics Data/YH38_240928_Kinematics_All.csv')
YH38_1_Y_L = pd.read_csv('Neural Data/YH38_240928_L_IRN_Single_Trial.csv')

YH38_1_X_R = pd.read_csv('Kinematics Data/YH38_240928_Kinematics_All.csv')
YH38_1_Y_R = pd.read_csv('Neural Data/YH38_240928_R_IRN_Single_Trial.csv')

# 09/29
YH38_2_X_L = pd.read_csv('Kinematics Data/YH38_240929_Kinematics_All.csv')
YH38_2_Y_L = pd.read_csv('Neural Data/YH38_240929_L_IRN_Single_Trial.csv')

YH38_2_X_R = pd.read_csv('Kinematics Data/YH38_240929_Kinematics_All.csv')
YH38_2_Y_R = pd.read_csv('Neural Data/YH38_240929_R_IRN_Single_Trial.csv')

# 09/30
YH38_3_X_L = pd.read_csv('Kinematics Data/YH38_240930_Kinematics_All.csv')
YH38_3_Y_L = pd.read_csv('Neural Data/YH38_240930_L_IRN_Single_Trial.csv')

YH38_3_X_R = pd.read_csv('Kinematics Data/YH38_240930_Kinematics_All.csv')
YH38_3_Y_R = pd.read_csv('Neural Data/YH38_240930_R_IRN_Single_Trial.csv')

# 10/01
YH38_4_X_L = pd.read_csv('Kinematics Data/YH38_241001_Kinematics_All.csv')
YH38_4_Y_L = pd.read_csv('Neural Data/YH38_241001_L_IRN_Single_Trial.csv')

YH38_4_X_R = pd.read_csv('Kinematics Data/YH38_241001_Kinematics_All.csv')
YH38_4_Y_R = pd.read_csv('Neural Data/YH38_241001_R_IRN_Single_Trial.csv')

# 10/02
YH38_5_X_L = pd.read_csv('Kinematics Data/YH38_241002_Kinematics_All.csv')
YH38_5_Y_L = pd.read_csv('Neural Data/YH38_241002_L_IRN_Single_Trial.csv')

YH38_5_X_R = pd.read_csv('Kinematics Data/YH38_241002_Kinematics_All.csv')
YH38_5_Y_R = pd.read_csv('Neural Data/YH38_241002_R_IRN_Single_Trial.csv')


# In[10]:


# Load Data - YH39

# 10/01
YH39_1_X_L = pd.read_csv('Kinematics Data/YH39_241001_Kinematics_All.csv')
YH39_1_Y_L = pd.read_csv('Neural Data/YH39_241001_L_IRN_Single_Trial.csv')

YH39_1_X_R = pd.read_csv('Kinematics Data/YH39_241001_Kinematics_All.csv')
YH39_1_Y_R = pd.read_csv('Neural Data/YH39_241001_R_IRN_Single_Trial.csv')

# 10/02
YH39_2_X_L = pd.read_csv('Kinematics Data/YH39_241002_Kinematics_All.csv')
YH39_2_Y_L = pd.read_csv('Neural Data/YH39_241002_L_IRN_Single_Trial.csv')

YH39_2_X_R = pd.read_csv('Kinematics Data/YH39_241002_Kinematics_All.csv')
YH39_2_Y_R = pd.read_csv('Neural Data/YH39_241002_R_IRN_Single_Trial.csv')

# 10/03
YH39_3_X_L = pd.read_csv('Kinematics Data/YH39_241003_Kinematics_All.csv')
YH39_3_Y_L = pd.read_csv('Neural Data/YH39_241003_L_IRN_Single_Trial.csv')

YH39_3_X_R = pd.read_csv('Kinematics Data/YH39_241003_Kinematics_All.csv')
YH39_3_Y_R = pd.read_csv('Neural Data/YH39_241003_R_IRN_Single_Trial.csv')

# # 10/04 
# YH39_4_X_L = pd.read_csv('Kinematics Data/YH39_241003_Kinematics_All.csv')
# YH39_4_Y_L = pd.read_csv('Neural Data/YH39_241003_L_IRN_Single_Trial.csv')

# YH39_4_X_R = pd.read_csv('Kinematics Data/YH39_241003_Kinematics_All.csv')
# YH39_4_Y_R = pd.read_csv('Neural Data/YH39_241003_R_IRN_Single_Trial.csv')


# In[11]:


# Data Preprocessing 
# PCA on Neural Data
# Z-scoring Kinematic Data
from scipy.stats import zscore 

def preprocess_data(X, Y, trial_len, PCs, session_id):

    
    # Drop Trial Type Column 
    if 'Trial Type' in X.columns: 
        X = X.drop(columns = ['Trial Type'])
    if 'Trial Type' in Y.columns: 
        Y = Y.drop(columns = ['Trial Type'])
    
    X = X.to_numpy()
    Y = Y.to_numpy()
    
    # Z - scoring Kinematic Features (X)
    X = zscore(X, axis = 0)
    
    # Perform SVD on the activity data to get principal components
    A = (Y - np.mean(Y, axis=0)).T
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    
    # Eigenvalues
    var_exp_pc = [(S[i]**2) / np.sum(S**2) for i in range(PCs)]
    
    # Perform Dimensionality Reduction
    S = np.diag(S)
    Y_pc = V[:, :PCs] @ S[:PCs, :PCs]
    
    # Calculate Variance in Each PC
    var_pc = [np.var(Y_pc[:,i]) for i in range(PCs)]
    

    # Session_ID array
    ID_array = session_id * np.ones((X.shape[0], 1))

    X_ID = np.concatenate((X, ID_array), axis = 1)
    Y_ID = np.concatenate((Y_pc, ID_array), axis = 1)

    X_reshaped = X_ID.reshape((X.shape[0] // trial_len, trial_len, X.shape[1] + 1))
    Y_reshaped = Y_ID.reshape((Y.shape[0] // trial_len, trial_len, PCs + 1)) 


    return X_reshaped, Y_reshaped, var_exp_pc 


# In[12]:


# Generate New X and Y 
trial_len = 350
PCs = 8

# --------------------------------------------------------- YH16 -----------------------------------------------------------
YH16_1_X_L_reshaped, YH16_1_Y_L_reshaped, YH16_var_exp_1_L = preprocess_data(YH16_1_X_L, YH16_1_Y_L, trial_len, PCs, 1)
YH16_1_X_R_reshaped, YH16_1_Y_R_reshaped, YH16_var_exp_1_R = preprocess_data(YH16_1_X_R, YH16_1_Y_R, trial_len, PCs, 2)

YH16_2_X_L_reshaped, YH16_2_Y_L_reshaped, YH16_var_exp_2_L = preprocess_data(YH16_2_X_L, YH16_2_Y_L, trial_len, PCs, 3)
YH16_2_X_R_reshaped, YH16_2_Y_R_reshaped, YH16_var_exp_2_R = preprocess_data(YH16_2_X_R, YH16_2_Y_R, trial_len, PCs, 4)

YH16_3_X_L_reshaped, YH16_3_Y_L_reshaped, YH16_var_exp_3_L = preprocess_data(YH16_3_X_L, YH16_3_Y_L, trial_len, PCs, 5)
YH16_3_X_R_reshaped, YH16_3_Y_R_reshaped, YH16_var_exp_3_R = preprocess_data(YH16_3_X_R, YH16_3_Y_R, trial_len, PCs, 6)

# --------------------------------------------------------- YH28 -----------------------------------------------------------
YH28_1_X_L_reshaped, YH28_1_Y_L_reshaped, YH28_var_exp_1_L = preprocess_data(YH28_1_X_L, YH28_1_Y_L, trial_len, PCs, 7)
YH28_1_X_R_reshaped, YH28_1_Y_R_reshaped, YH28_var_exp_1_R = preprocess_data(YH28_1_X_R, YH28_1_Y_R, trial_len, PCs, 8)

YH28_2_X_L_reshaped, YH28_2_Y_L_reshaped, YH28_var_exp_2_L = preprocess_data(YH28_2_X_L, YH28_2_Y_L, trial_len, PCs, 9)
YH28_2_X_R_reshaped, YH28_2_Y_R_reshaped, YH28_var_exp_2_R = preprocess_data(YH28_2_X_R, YH28_2_Y_R, trial_len, PCs, 10)

YH28_3_X_L_reshaped, YH28_3_Y_L_reshaped, YH28_var_exp_3_L = preprocess_data(YH28_3_X_L, YH28_3_Y_L, trial_len, PCs, 11)
YH28_3_X_R_reshaped, YH28_3_Y_R_reshaped, YH28_var_exp_3_R = preprocess_data(YH28_3_X_R, YH28_3_Y_R, trial_len, PCs, 12)

YH28_4_X_L_reshaped, YH28_4_Y_L_reshaped, YH28_var_exp_4_L = preprocess_data(YH28_4_X_L, YH28_4_Y_L, trial_len, PCs, 13)
YH28_4_X_R_reshaped, YH28_4_Y_R_reshaped, YH28_var_exp_4_R = preprocess_data(YH28_4_X_R, YH28_4_Y_R, trial_len, PCs, 14)


# --------------------------------------------------------- YH31 -----------------------------------------------------------
YH31_1_X_L_reshaped, YH31_1_Y_L_reshaped, YH31_var_exp_1_L = preprocess_data(YH31_1_X_L, YH31_1_Y_L, trial_len, PCs, 15)
YH31_1_X_R_reshaped, YH31_1_Y_R_reshaped, YH31_var_exp_1_R = preprocess_data(YH31_1_X_R, YH31_1_Y_R, trial_len, PCs, 16)

YH31_2_X_L_reshaped, YH31_2_Y_L_reshaped, YH31_var_exp_2_L = preprocess_data(YH31_2_X_L, YH31_2_Y_L, trial_len, PCs, 17)
YH31_2_X_R_reshaped, YH31_2_Y_R_reshaped, YH31_var_exp_2_R = preprocess_data(YH31_2_X_R, YH31_2_Y_R, trial_len, PCs, 18)

YH31_3_X_L_reshaped, YH31_3_Y_L_reshaped, YH31_var_exp_3_L = preprocess_data(YH31_3_X_L, YH31_3_Y_L, trial_len, PCs, 19)
YH31_3_X_R_reshaped, YH31_3_Y_R_reshaped, YH31_var_exp_3_R = preprocess_data(YH31_3_X_R, YH31_3_Y_R, trial_len, PCs, 20)

YH31_4_X_L_reshaped, YH31_4_Y_L_reshaped, YH31_var_exp_4_L = preprocess_data(YH31_4_X_L, YH31_4_Y_L, trial_len, PCs, 21)
YH31_4_X_R_reshaped, YH31_4_Y_R_reshaped, YH31_var_exp_4_R = preprocess_data(YH31_4_X_R, YH31_4_Y_R, trial_len, PCs, 22)

YH31_5_X_L_reshaped, YH31_5_Y_L_reshaped, YH31_var_exp_5_L = preprocess_data(YH31_5_X_L, YH31_5_Y_L, trial_len, PCs, 23)
YH31_5_X_R_reshaped, YH31_5_Y_R_reshaped, YH31_var_exp_5_R = preprocess_data(YH31_5_X_R, YH31_5_Y_R, trial_len, PCs, 24)

# --------------------------------------------------------- YH32 -----------------------------------------------------------
YH32_1_X_L_reshaped, YH32_1_Y_L_reshaped, YH32_var_exp_1_L = preprocess_data(YH32_1_X_L, YH32_1_Y_L, trial_len, PCs, 25)

YH32_2_X_R_reshaped, YH32_2_Y_R_reshaped, YH32_var_exp_2_R = preprocess_data(YH32_2_X_R, YH32_2_Y_R, trial_len, PCs, 26)

YH32_3_X_R_reshaped, YH32_3_Y_R_reshaped, YH32_var_exp_3_R = preprocess_data(YH32_3_X_R, YH32_3_Y_R, trial_len, PCs, 27)

# --------------------------------------------------------- YH32 -----------------------------------------------------------
YH38_1_X_L_reshaped, YH38_1_Y_L_reshaped, YH38_var_exp_1_L = preprocess_data(YH38_1_X_L, YH38_1_Y_L, trial_len, PCs, 28)
YH38_1_X_R_reshaped, YH38_1_Y_R_reshaped, YH38_var_exp_1_R = preprocess_data(YH38_1_X_R, YH38_1_Y_R, trial_len, PCs, 29)

YH38_2_X_L_reshaped, YH38_2_Y_L_reshaped, YH38_var_exp_2_L = preprocess_data(YH38_2_X_L, YH38_2_Y_L, trial_len, PCs, 30)
YH38_2_X_R_reshaped, YH38_2_Y_R_reshaped, YH38_var_exp_2_R = preprocess_data(YH38_2_X_R, YH38_2_Y_R, trial_len, PCs, 31)

YH38_3_X_L_reshaped, YH38_3_Y_L_reshaped, YH38_var_exp_3_L = preprocess_data(YH38_3_X_L, YH38_3_Y_L, trial_len, PCs, 32)
YH38_3_X_R_reshaped, YH38_3_Y_R_reshaped, YH38_var_exp_3_R = preprocess_data(YH38_3_X_R, YH38_3_Y_R, trial_len, PCs, 33)

YH38_4_X_L_reshaped, YH38_4_Y_L_reshaped, YH38_var_exp_4_L = preprocess_data(YH38_4_X_L, YH38_4_Y_L, trial_len, PCs, 34)
YH38_4_X_R_reshaped, YH38_4_Y_R_reshaped, YH38_var_exp_4_R = preprocess_data(YH38_4_X_R, YH38_4_Y_R, trial_len, PCs, 35)

YH38_5_X_L_reshaped, YH38_5_Y_L_reshaped, YH38_var_exp_5_L = preprocess_data(YH38_5_X_L, YH38_5_Y_L, trial_len, PCs, 36)
YH38_5_X_R_reshaped, YH38_5_Y_R_reshaped, YH38_var_exp_5_R = preprocess_data(YH38_5_X_R, YH38_5_Y_R, trial_len, PCs, 37)

# --------------------------------------------------------- YH39 -----------------------------------------------------------
YH39_1_X_L_reshaped, YH39_1_Y_L_reshaped, YH39_var_exp_1_L = preprocess_data(YH39_1_X_L, YH39_1_Y_L, trial_len, PCs, 38)
YH39_1_X_R_reshaped, YH39_1_Y_R_reshaped, YH39_var_exp_1_R = preprocess_data(YH39_1_X_R, YH39_1_Y_R, trial_len, PCs, 39)

YH39_2_X_L_reshaped, YH39_2_Y_L_reshaped, YH39_var_exp_2_L = preprocess_data(YH39_2_X_L, YH39_2_Y_L, trial_len, PCs, 40)
YH39_2_X_R_reshaped, YH39_2_Y_R_reshaped, YH39_var_exp_2_R = preprocess_data(YH39_2_X_R, YH39_2_Y_R, trial_len, PCs, 41)

YH39_3_X_L_reshaped, YH39_3_Y_L_reshaped, YH39_var_exp_3_L = preprocess_data(YH39_3_X_L, YH39_3_Y_L, trial_len, PCs, 42)
YH39_3_X_R_reshaped, YH39_3_Y_R_reshaped, YH39_var_exp_3_R = preprocess_data(YH39_3_X_R, YH39_3_Y_R, trial_len, PCs, 43)


# In[13]:


# Test and Train split for Different Sessions
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split

# Function for creating Data Loaders
def create_dataloader(X, Y, batch_size=20):
    data = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return dataloader


# In[14]:


# Create Loaders for YH16 
loader_YH16_1_L = create_dataloader(YH16_1_X_L_reshaped, YH16_1_Y_L_reshaped)
loader_YH16_1_R = create_dataloader(YH16_1_X_R_reshaped, YH16_1_Y_R_reshaped)
loader_YH16_2_L = create_dataloader(YH16_2_X_L_reshaped, YH16_2_Y_L_reshaped)
loader_YH16_2_R = create_dataloader(YH16_2_X_R_reshaped, YH16_2_Y_R_reshaped)
loader_YH16_3_L = create_dataloader(YH16_3_X_L_reshaped, YH16_3_Y_L_reshaped)
loader_YH16_3_R = create_dataloader(YH16_3_X_R_reshaped, YH16_3_Y_R_reshaped)


# In[15]:


# Create Loaders for YH28
loader_YH28_1_L = create_dataloader(YH28_1_X_L_reshaped, YH28_1_Y_L_reshaped)
loader_YH28_1_R = create_dataloader(YH28_1_X_R_reshaped, YH28_1_Y_R_reshaped)
loader_YH28_2_L = create_dataloader(YH28_2_X_L_reshaped, YH28_2_Y_L_reshaped)
loader_YH28_2_R = create_dataloader(YH28_2_X_R_reshaped, YH28_2_Y_R_reshaped)
loader_YH28_3_L = create_dataloader(YH28_3_X_L_reshaped, YH28_3_Y_L_reshaped)
loader_YH28_3_R = create_dataloader(YH28_3_X_R_reshaped, YH28_3_Y_R_reshaped)
loader_YH28_4_L = create_dataloader(YH28_4_X_L_reshaped, YH28_4_Y_L_reshaped)
loader_YH28_4_R = create_dataloader(YH28_4_X_R_reshaped, YH28_4_Y_R_reshaped)


# In[16]:


# Create Loaders for YH31
loader_YH31_1_L = create_dataloader(YH31_1_X_L_reshaped, YH31_1_Y_L_reshaped)
loader_YH31_1_R = create_dataloader(YH31_1_X_R_reshaped, YH31_1_Y_R_reshaped)
loader_YH31_2_L = create_dataloader(YH31_2_X_L_reshaped, YH31_2_Y_L_reshaped)
loader_YH31_2_R = create_dataloader(YH31_2_X_R_reshaped, YH31_2_Y_R_reshaped)
loader_YH31_3_L = create_dataloader(YH31_3_X_L_reshaped, YH31_3_Y_L_reshaped)
loader_YH31_3_R = create_dataloader(YH31_3_X_R_reshaped, YH31_3_Y_R_reshaped)
loader_YH31_4_L = create_dataloader(YH31_4_X_L_reshaped, YH31_4_Y_L_reshaped)
loader_YH31_4_R = create_dataloader(YH31_4_X_R_reshaped, YH31_4_Y_R_reshaped)
loader_YH31_5_L = create_dataloader(YH31_5_X_L_reshaped, YH31_5_Y_L_reshaped)
loader_YH31_5_R = create_dataloader(YH31_5_X_R_reshaped, YH31_5_Y_R_reshaped)


# In[17]:


# Create Loaders for YH32 
loader_YH32_1_L = create_dataloader(YH32_1_X_L_reshaped, YH32_1_Y_L_reshaped)
loader_YH32_2_R = create_dataloader(YH32_2_X_R_reshaped, YH32_2_Y_R_reshaped)
loader_YH32_3_R = create_dataloader(YH32_3_X_R_reshaped, YH32_3_Y_R_reshaped)


# In[18]:


# Create Loaders for YH38
loader_YH38_1_L = create_dataloader(YH38_1_X_L_reshaped, YH38_1_Y_L_reshaped)
loader_YH38_1_R = create_dataloader(YH38_1_X_R_reshaped, YH38_1_Y_R_reshaped)
loader_YH38_2_L = create_dataloader(YH38_2_X_L_reshaped, YH38_2_Y_L_reshaped)
loader_YH38_2_R = create_dataloader(YH38_2_X_R_reshaped, YH38_2_Y_R_reshaped)
loader_YH38_3_L = create_dataloader(YH38_3_X_L_reshaped, YH38_3_Y_L_reshaped)
loader_YH38_3_R = create_dataloader(YH38_3_X_R_reshaped, YH38_3_Y_R_reshaped)
loader_YH38_4_L = create_dataloader(YH38_4_X_L_reshaped, YH38_4_Y_L_reshaped)
loader_YH38_4_R = create_dataloader(YH38_4_X_R_reshaped, YH38_4_Y_R_reshaped)
loader_YH38_5_L = create_dataloader(YH38_5_X_L_reshaped, YH38_5_Y_L_reshaped)
loader_YH38_5_R = create_dataloader(YH38_5_X_R_reshaped, YH38_5_Y_R_reshaped)


# In[19]:


# Create Loaders for YH39
loader_YH39_1_L = create_dataloader(YH39_1_X_L_reshaped, YH39_1_Y_L_reshaped)
loader_YH39_1_R = create_dataloader(YH39_1_X_R_reshaped, YH39_1_Y_R_reshaped)
loader_YH39_2_L = create_dataloader(YH39_2_X_L_reshaped, YH39_2_Y_L_reshaped)
loader_YH39_2_R = create_dataloader(YH39_2_X_R_reshaped, YH39_2_Y_R_reshaped)
loader_YH39_3_L = create_dataloader(YH39_3_X_L_reshaped, YH39_3_Y_L_reshaped)
loader_YH39_3_R = create_dataloader(YH39_3_X_R_reshaped, YH39_3_Y_R_reshaped)


# In[20]:


# Extract Batches Function
def extract_batches(loader):
    batches = []
    for batch in loader: 
        batches.append(batch)
    return batches


# =========================================================== MODEL ========================================================

# In[58]:


import torch.nn as nn 

class LSTM_model(nn.Module):
    def __init__(self, input_dim, hidden_dim1, output_dim, num_layers):
        super(LSTM_model, self).__init__()
        self.hidden_dim1 = hidden_dim1
        self.num_layers = num_layers
        # self.test_session_id = test_session_id  

        self.lstm = nn.LSTM(input_dim, hidden_dim1, num_layers, batch_first=True, bias=True, dropout=0.3)
        self.relu = nn.ReLU()

        # Linear layers for each session
        self.fc_layers = nn.ModuleDict({
            '1': nn.Linear(hidden_dim1, output_dim),
            '2': nn.Linear(hidden_dim1, output_dim),
            '3': nn.Linear(hidden_dim1, output_dim),
            '4': nn.Linear(hidden_dim1, output_dim),
            '5': nn.Linear(hidden_dim1, output_dim), 
            '6': nn.Linear(hidden_dim1, output_dim), 
            '7': nn.Linear(hidden_dim1, output_dim),
            '8': nn.Linear(hidden_dim1, output_dim),
            '9': nn.Linear(hidden_dim1, output_dim),
            '10': nn.Linear(hidden_dim1, output_dim),
            '11': nn.Linear(hidden_dim1, output_dim),
            '12': nn.Linear(hidden_dim1, output_dim),
            '13': nn.Linear(hidden_dim1, output_dim),
            '14': nn.Linear(hidden_dim1, output_dim), 
            # '15': nn.Linear(hidden_dim1, output_dim), 
            # '16': nn.Linear(hidden_dim1, output_dim),
            # '17': nn.Linear(hidden_dim1, output_dim),
            # '18': nn.Linear(hidden_dim1, output_dim),
            # '19': nn.Linear(hidden_dim1, output_dim),
            # '20': nn.Linear(hidden_dim1, output_dim),
            # '21': nn.Linear(hidden_dim1, output_dim),
            # '22': nn.Linear(hidden_dim1, output_dim),
            # '23': nn.Linear(hidden_dim1, output_dim),
            # '24': nn.Linear(hidden_dim1, output_dim),
            '25': nn.Linear(hidden_dim1, output_dim),
            '26': nn.Linear(hidden_dim1, output_dim),
            '27': nn.Linear(hidden_dim1, output_dim),
            '28': nn.Linear(hidden_dim1, output_dim),
            '29': nn.Linear(hidden_dim1, output_dim),
            '30': nn.Linear(hidden_dim1, output_dim),
            '31': nn.Linear(hidden_dim1, output_dim),
            '32': nn.Linear(hidden_dim1, output_dim),
            '33': nn.Linear(hidden_dim1, output_dim),
            '34': nn.Linear(hidden_dim1, output_dim),
            '35': nn.Linear(hidden_dim1, output_dim),
            '36': nn.Linear(hidden_dim1, output_dim),
            '37': nn.Linear(hidden_dim1, output_dim),
            '38': nn.Linear(hidden_dim1, output_dim),
            '39': nn.Linear(hidden_dim1, output_dim),
            '40': nn.Linear(hidden_dim1, output_dim),
            '41': nn.Linear(hidden_dim1, output_dim),
            '42': nn.Linear(hidden_dim1, output_dim),
            '43': nn.Linear(hidden_dim1, output_dim),
        })

    def forward(self, x):
        # print(x.shape)
        session_id = int(x[0, :, -1][0].item())
        # print(session_id)
        x = x[:, :, :-1]

        # LSTM forward pass
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim1).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim1).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.relu(out)

        # Apply the corresponding linear layer
        if str(session_id) in self.fc_layers:
            out = self.fc_layers[str(session_id)](out)
        else:
            print(f"Unexpected Session ID: {session_id}")
            return out

        return out


# In[59]:


# Set LSTM Model Parameters

# Parameters
input_dim = 55  # Input dimension to LSTM
hidden_dim1 = 128  # Hidden dimension of LSTM
output_dim = PCs  # Number of principal components (PCs)
num_layers = 3  # LSTM Layers


# In[60]:


# Model Optimizer and Loss Function 

import torch.optim as optim

Loss_fcn = nn.MSELoss()

# R-squared Function 
from sklearn.metrics import r2_score

import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Suppress the specific warning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


# In[61]:


# Set Train and Test Loaders (Training on Bad Sessions)

model = LSTM_model(input_dim, hidden_dim1, output_dim, num_layers)
model.to(device)

# Test on YH16
# train_loaders = [
#     loader_YH28_1_L, loader_YH28_1_R, loader_YH28_2_L, loader_YH28_2_R,
#     loader_YH28_3_L, loader_YH28_3_R, loader_YH28_4_L, loader_YH28_4_R,
#     loader_YH31_1_L, loader_YH31_1_R, loader_YH31_2_L, loader_YH31_2_R,
#     loader_YH31_3_L, loader_YH31_3_R, loader_YH31_4_L, loader_YH31_4_R,
#     loader_YH31_5_L, loader_YH31_5_R, loader_YH32_1_L, loader_YH32_2_R, 
#     loader_YH32_3_R, 
#     loader_YH38_1_L, loader_YH38_1_R, loader_YH38_2_L, loader_YH38_2_R,
#     loader_YH38_3_L, loader_YH38_3_R, loader_YH38_4_L, loader_YH38_4_R,
#     loader_YH38_5_L, loader_YH38_5_R, 
#     loader_YH39_1_L, loader_YH39_1_R, loader_YH39_2_L, loader_YH39_2_R, 
#     loader_YH39_3_L, loader_YH39_3_R
# ]

# test_loaders = [
#     loader_YH16_1_L, loader_YH16_1_R, loader_YH16_2_L, loader_YH16_2_R,
#     loader_YH16_3_L, loader_YH16_3_R
# ]

# # Test on YH28
# train_loaders = [
#     loader_YH16_1_L, loader_YH16_1_R, loader_YH16_2_L, loader_YH16_2_R,
#     loader_YH16_3_L, loader_YH16_3_R,
#     loader_YH31_1_L, loader_YH31_1_R, loader_YH31_2_L, loader_YH31_2_R,
#     loader_YH31_3_L, loader_YH31_3_R, loader_YH31_4_L, loader_YH31_4_R,
#     loader_YH31_5_L, loader_YH31_5_R, loader_YH32_1_L, loader_YH32_2_R, 
#     loader_YH32_3_R, 
#     loader_YH38_1_L, loader_YH38_1_R, loader_YH38_2_L, loader_YH38_2_R,
#     loader_YH38_3_L, loader_YH38_3_R, loader_YH38_4_L, loader_YH38_4_R,
#     loader_YH38_5_L, loader_YH38_5_R, 
#     loader_YH39_1_L, loader_YH39_1_R, loader_YH39_2_L, loader_YH39_2_R, 
#     loader_YH39_3_L, loader_YH39_3_R
# ]

# test_loaders = [
#     loader_YH28_1_L, loader_YH28_1_R, loader_YH28_2_L, loader_YH28_2_R,
#     loader_YH28_3_L, loader_YH28_3_R, loader_YH28_4_L, loader_YH28_4_R
# ]

# # Test on YH31
train_loaders = [
    loader_YH16_1_L, loader_YH16_1_R, loader_YH16_2_L, loader_YH16_2_R,
    loader_YH16_3_L, loader_YH16_3_R,
    loader_YH28_1_L, loader_YH28_1_R, loader_YH28_2_L, loader_YH28_2_R,
    loader_YH28_3_L, loader_YH28_3_R, loader_YH28_4_L, loader_YH28_4_R,
    loader_YH32_1_L, loader_YH32_2_R, loader_YH32_3_R, 
    loader_YH38_1_L, loader_YH38_1_R, loader_YH38_2_L, loader_YH38_2_R,
    loader_YH38_3_L, loader_YH38_3_R, loader_YH38_4_L, loader_YH38_4_R,
    loader_YH38_5_L, loader_YH38_5_R, 
    loader_YH39_1_L, loader_YH39_1_R, loader_YH39_2_L, loader_YH39_2_R, 
    loader_YH39_3_L, loader_YH39_3_R
]

test_loaders = [
    loader_YH31_1_L, loader_YH31_1_R, loader_YH31_2_L, loader_YH31_2_R,
    loader_YH31_3_L, loader_YH31_3_R, loader_YH31_4_L, loader_YH31_4_R,
    loader_YH31_5_L, loader_YH31_5_R
]

# Test on YH32
# train_loaders = [
#     loader_YH16_1_L, loader_YH16_1_R, loader_YH16_2_L, loader_YH16_2_R,
#     loader_YH16_3_L, loader_YH16_3_R,
#     loader_YH28_1_L, loader_YH28_1_R, loader_YH28_2_L, loader_YH28_2_R,
#     loader_YH28_3_L, loader_YH28_3_R, loader_YH28_4_L, loader_YH28_4_R,
#     loader_YH31_1_L, loader_YH31_1_R, loader_YH31_2_L, loader_YH31_2_R,
#     loader_YH31_3_L, loader_YH31_3_R, loader_YH31_4_L, loader_YH31_4_R,
#     loader_YH31_5_L, loader_YH31_5_R,
#     loader_YH38_1_L, loader_YH38_1_R, loader_YH38_2_L, loader_YH38_2_R,
#     loader_YH38_3_L, loader_YH38_3_R, loader_YH38_4_L, loader_YH38_4_R,
#     loader_YH38_5_L, loader_YH38_5_R, 
#     loader_YH39_1_L, loader_YH39_1_R, loader_YH39_2_L, loader_YH39_2_R, 
#     loader_YH39_3_L, loader_YH39_3_R
# ]

# test_loaders = [
#     loader_YH32_1_L, loader_YH32_2_R, loader_YH32_3_R
# ]

# # Test on YH38
# train_loaders = [
#     loader_YH16_1_L, loader_YH16_1_R, loader_YH16_2_L, loader_YH16_2_R,
#     loader_YH16_3_L, loader_YH16_3_R,
#     loader_YH28_1_L, loader_YH28_1_R, loader_YH28_2_L, loader_YH28_2_R,
#     loader_YH28_3_L, loader_YH28_3_R, loader_YH28_4_L, loader_YH28_4_R,
#     loader_YH31_1_L, loader_YH31_1_R, loader_YH31_2_L, loader_YH31_2_R,
#     loader_YH31_3_L, loader_YH31_3_R, loader_YH31_4_L, loader_YH31_4_R,
#     loader_YH31_5_L, loader_YH31_5_R,
#     loader_YH32_1_L, loader_YH32_2_R, loader_YH32_3_R,
#     loader_YH39_1_L, loader_YH39_1_R, loader_YH39_2_L, loader_YH39_2_R, 
#     loader_YH39_3_L, loader_YH39_3_R
# ]

# test_loaders = [
#     loader_YH38_1_L, loader_YH38_1_R, loader_YH38_2_L, loader_YH38_2_R,
#     loader_YH38_3_L, loader_YH38_3_R, loader_YH38_4_L, loader_YH38_4_R,
#     loader_YH38_5_L, loader_YH38_5_R
# ]

# # Test on YH39
# train_loaders = [
#     loader_YH16_1_L, loader_YH16_1_R, loader_YH16_2_L, loader_YH16_2_R,
#     loader_YH16_3_L, loader_YH16_3_R,
#     loader_YH28_1_L, loader_YH28_1_R, loader_YH28_2_L, loader_YH28_2_R,
#     loader_YH28_3_L, loader_YH28_3_R, loader_YH28_4_L, loader_YH28_4_R,
#     loader_YH31_1_L, loader_YH31_1_R, loader_YH31_2_L, loader_YH31_2_R,
#     loader_YH31_3_L, loader_YH31_3_R, loader_YH31_4_L, loader_YH31_4_R,
#     loader_YH31_5_L, loader_YH31_5_R,
#     loader_YH32_1_L, loader_YH32_2_R, loader_YH32_3_R,
#     loader_YH38_1_L, loader_YH38_1_R, loader_YH38_2_L, loader_YH38_2_R,
#     loader_YH38_3_L, loader_YH38_3_R, loader_YH38_4_L, loader_YH38_4_R,
#     loader_YH38_5_L, loader_YH38_5_R
# ]

# test_loaders = [
#     loader_YH39_1_L, loader_YH39_1_R, loader_YH39_2_L, loader_YH39_2_R,
#     loader_YH39_3_L, loader_YH39_3_R
# ]

all_batches = []
for loader in train_loaders: 
    all_batches.append(extract_batches(loader))

combined_batches = [batch for session_batches in all_batches for batch in session_batches]
random.shuffle(combined_batches)
print(len(combined_batches))

train_loader = DataLoader(combined_batches, batch_size=1, shuffle=False)

print(len(train_loaders) + len(test_loaders))


# =============================================== Model Training ====================================================

# In[66]:


# Model Training - Cross Animal --- Train on three animals ---> Test on the fourth
import time 

start_time = time.time()

model = LSTM_model(input_dim, hidden_dim1, output_dim, num_layers)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

best_r2 = -float('inf') 
model_save_path = f'Blocked CV models (ALL QCED)/Model_tested_on_TRIAL.pth'
num_epochs = 1500

train_Rsquared = []

print(f'Training Started')

for epoch in range(num_epochs):
    model.train()
    train_r2 = 0.0
    pc_r2 = torch.zeros(output_dim).to(device)

    for X_batch, Y_batch in train_loader:
        features = X_batch.to(device).squeeze(0)
        target = Y_batch.to(device).squeeze(0)
        optimizer.zero_grad()

        target = target[:, :, :-1]
        outputs = model(features)
        # print(target.shape)
        # print(outputs.shape)

        loss = Loss_fcn(outputs, target)
        loss.backward()
        optimizer.step()

        r2 = r2_score(target.cpu().numpy().reshape(-1, target.shape[-1]), 
                      outputs.cpu().detach().numpy().reshape(-1, outputs.shape[-1]))
        r2_pc = torch.tensor([r2_score(target[:, :, i].cpu().numpy(), 
                                       outputs[:, :, i].cpu().detach().numpy()) for i in range(output_dim)]).to(device)

        train_r2 += r2
        pc_r2 += r2_pc

    pc_r2 /= len(train_loader.dataset)
    train_r2 /= len(train_loader.dataset)
    train_Rsquared.append(train_r2)

    if train_r2 > best_r2:
        best_r2 = train_r2
        torch.save(model.state_dict(), model_save_path)

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], R2: {train_r2:.4f}')

end_time = time.time()
print(f'Training Done {(end_time - start_time) / 60 :.2f} Minutes')


# In[67]:


# Training Performance: 
plt.plot(range(num_epochs), train_Rsquared, color = 'black')
plt.xlabel('Training Epoch')
plt.ylabel('R²')
plt.ylim((0,1))
plt.xlim((0, num_epochs))


# =============================================== Model Testing ====================================================

# In[63]:


# Model Testing --> Cross-Animal Testing 

YH31_perf = []
YH31_sizes = []

for test_loader in test_loaders[:8]:
    model.load_state_dict(torch.load(f'Blocked CV models (ALL QCED)/Model_tested_on_YH31_QC.pth'))
    model.eval()

    ID = int(test_loader.dataset[9][0][0, -1])
    print(f'Test_Loader_ID:', ID)

    all_targets = test_loader.dataset[:][1][:, :, :-1]
    all_features = (test_loader.dataset[:][0][:, :, :]).to(device)

    with torch.no_grad():
        all_outputs = model(all_features)
        print(all_outputs.shape)

    X_regression = all_outputs.cpu().numpy().reshape(-1, all_outputs.shape[-1])
    Y_regression = all_targets.cpu().numpy().reshape(-1, all_targets.shape[-1])

    BETA_OLS = np.linalg.lstsq(X_regression, Y_regression, rcond=None)
    print(BETA_OLS[0].shape)

    Y_hat = X_regression @ BETA_OLS[0]
    r2_pc_reg = [r2_score(Y_regression[:, i], Y_hat[:, i]) for i in range(PCs)]
    Y_hat_reshaped = Y_hat.reshape(all_targets.shape[0], all_targets.shape[1], all_targets.shape[2])
    print(Y_hat_reshaped.shape[0])

    YH31_sizes.append(Y_hat_reshaped.shape[0])
    YH31_perf.append(r2_pc_reg)
    print(f"Completed Evaluation for Loader {ID} - {test_loader}")
    print('')


# ========================================= Model Performance Plots ==========================================

# In[30]:


# Box and Whiskers Plot
import seaborn as sns 

perf_np = np.array(YH16_perf)
np.set_printoptions(precision=3)
print(perf_np)
plt.figure(figsize = (10,6))
sns.boxplot(data = perf_np)

plt.xlabel('PCs')
plt.ylabel('R²')
plt.title('Model Performance on YH16')
plt.ylim((0,1))

plt.xticks(range(8), labels = range(1,9))
plt.show()


# In[52]:


# Weighted Average Bar Plot - CV 
Weighted_Results_16 = np.sum([np.array(YH16_perf[i]) * (YH16_sizes[i]/np.sum(YH16_sizes)) for i in range(len(YH16_perf))], axis = 0)
print(Weighted_Results_16)

plt.bar(range(1,1+(PCs)), Weighted_Results_16, color = 'blue')
plt.xlabel('PCs')
plt.ylabel('R²')
plt.ylim((0,1))
plt.title('Model Performance on YH16')


# ================================================ Save Performances =============================================

# In[ ]:


# Save Fold Performance
Performances = [np.array(YH16_perf), np.array(YH28_perf), np.array(YH31_perf), np.array(YH32_perf), np.array(YH38_perf), np.array(YH39_perf)]


# In[ ]:


# Plot Results across all CVs

all_performances = np.vstack((np.array(YH16_perf),np.array(YH28_perf), np.array(YH31_perf), np.array(YH32_perf), np.array(YH38_perf), np.array(YH39_perf)))
plt.figure(figsize = (10,6))
sns.boxplot(data = all_performances)

plt.xlabel('PCs')
plt.ylabel('R²')
plt.ylim((0,1))

plt.xticks(range(8), labels = range(1,9))
plt.show()


# In[ ]:


df = pd.DataFrame(all_performances)
stats = df.describe()
print(stats)


# In[ ]:


# Model Performance Bar Chart 
Performances = [Weighted_Results_16, Weighted_Results_28, Weighted_Results_31, Weighted_Results_32, Weighted_Results_38, Weighted_Results_39]
Trial_sessions = [YH16_sizes, YH28_sizes, YH31_sizes, YH32_sizes, YH38_sizes, YH39_sizes]
Number_of_trials = np.sum([np.sum(Trial_sessions[i]) for i in range(6)])
print(Number_of_trials)
Weighted_Results_All_Folds = np.sum([Performances[i] * (np.sum(Trial_sessions[i])/ Number_of_trials) for i in range(6)], axis = 0)

print(Weighted_Results_All_Folds)

plt.bar(range(1,1+(PCs)), Weighted_Results_All_Folds, color = 'black')
plt.xlabel('PCs')
plt.ylabel('R²')
plt.ylim((0,1))
plt.title('Model Performance')


# In[56]:


# HeatMap 

# Heat Map for PCs

PC = 2

actual_vals = all_targets[:, :, PC - 1].cpu().numpy()
output_vals = Y_hat_reshaped[:, :, PC - 1] 
print(actual_vals.shape)

full_seq = np.hstack((actual_vals, output_vals, np.abs(actual_vals - output_vals)))
print(full_seq.shape)

# Plot HeatMap
plt.figure(figsize=(12, 6))
plt.imshow(full_seq, aspect='auto', cmap = 'jet', origin='upper', vmin = - 300, vmax = 350)
plt.colorbar()
plt.xlabel('Time Points', fontweight='bold', fontsize=12)
plt.axvline(x = 350, color = 'k', linestyle = '--', linewidth = 3)
plt.axvline(x = 700, color = 'k', linestyle = '--', linewidth = 3)
plt.xlim((0, 1050))
plt.ylabel('Trials', fontweight='bold', fontsize=12)
plt.title(f'PC{PC} - R² = {YH39_perf[5][PC-1]:.2f}\n YH39 - Session 3_R', fontsize=20)

plt.show()


# In[64]:


# Print Variance Explained for each Animal
YH16_var_exp = [YH16_var_exp_1_L, YH16_var_exp_1_R, YH16_var_exp_2_L, YH16_var_exp_2_R, YH16_var_exp_3_L, YH16_var_exp_3_R]
YH28_var_exp = [YH28_var_exp_1_L, YH28_var_exp_1_R, YH28_var_exp_2_L, YH28_var_exp_2_R, YH28_var_exp_3_L, YH28_var_exp_3_R, YH28_var_exp_4_L, YH28_var_exp_4_R]
YH31_var_exp = [YH31_var_exp_1_L, YH31_var_exp_1_R, YH31_var_exp_2_L, YH31_var_exp_2_R, YH31_var_exp_3_L, YH31_var_exp_3_R, YH31_var_exp_4_L, YH31_var_exp_4_R, YH31_var_exp_5_L, YH31_var_exp_5_R]
YH32_var_exp = [YH32_var_exp_1_L, YH32_var_exp_2_R, YH32_var_exp_3_R]
YH38_var_exp = [YH38_var_exp_1_L, YH38_var_exp_1_R, YH38_var_exp_2_L, YH38_var_exp_2_R, YH38_var_exp_3_L, YH38_var_exp_3_R, YH38_var_exp_4_L, YH38_var_exp_4_R, YH38_var_exp_5_L, YH38_var_exp_5_R]
YH39_var_exp = [YH39_var_exp_1_L, YH39_var_exp_1_R, YH39_var_exp_2_L, YH39_var_exp_2_R, YH39_var_exp_3_L, YH39_var_exp_3_R]

# Labels for each session
# labels = ["YH16 - Session 1 Left", "YH16 - Session 1 Right", 
#           "YH16 - Session 2 Left", "YH16 - Session 2 Right", 
#           "YH16 - Session 3 Left", "YH16 - Session 3 Right"]

print('YH31 Variance Explained\n')

np.set_printoptions(precision=3)

for var_exp in YH31_var_exp:
    print(np.array(var_exp))

print()
    
for YH_perf in YH31_perf: 
    print(np.array(YH_perf))


# In[104]:


# Variance explained by Model 

Animal = ['YH16', 'YH28', 'YH31', 'YH32', 'YH38', 'YH39']
# print(YH28_perf[3])
# print(np.sum(YH28_var_exp[3]))
# print(np.dot(YH28_perf[3], YH28_var_exp[3]))


Var_exp_by_model_28 = [np.dot(YH28_perf[i], YH28_var_exp[i]) for i in range(len(YH28_perf))]

print(f'Variance Explained by Model for {Animal[1]}')
print()
print(Var_exp_by_model_28)


# In[84]:


# Concatenate All into one List 

Var_exp_MODEL = np.hstack((Var_exp_by_model_16, Var_exp_by_model_28, Var_exp_by_model_31, Var_exp_by_model_32, Var_exp_by_model_38, Var_exp_by_model_39))

print(len(Var_exp_MODEL))

# Save as CSV 
df = pd.DataFrame(Var_exp_MODEL)
df.to_csv('Variance_exp_BlockedCV_not_QC.csv', index = False)

print('List Saved Successfully')


# In[ ]:




