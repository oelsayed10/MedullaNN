# Facemap Preprocessing - Omar

# Preprocessing 

# Import Libraries 
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd
import torch 
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import random


# Load Data - YH16

# 03/06
YH16_1_X_L = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH16_240306_Kinematics_All.csv')
YH16_1_Y_L = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH16_240306_L_IRN_Single_Trial.csv')

YH16_1_X_R = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH16_240306_Kinematics_All.csv')
YH16_1_Y_R = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH16_240306_R_IRN_Single_Trial.csv')

# 03/07
YH16_2_X_L = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH16_240307_Kinematics_All.csv')
YH16_2_Y_L = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH16_240307_L_IRN_Single_Trial.csv')

YH16_2_X_R = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH16_240307_Kinematics_All.csv')
YH16_2_Y_R = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH16_240307_R_IRN_Single_Trial.csv')

# 03/08
YH16_3_X_L = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH16_240308_Kinematics_All.csv')
YH16_3_Y_L = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH16_240308_L_IRN_Single_Trial.csv')

YH16_3_X_R = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH16_240308_Kinematics_All.csv')
YH16_3_Y_R = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH16_240308_R_IRN_Single_Trial.csv')




# Load Data - YH28

# 06/28
YH28_1_X_L = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH28_240628_Kinematics_All.csv')
YH28_1_Y_L = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH28_240628_L_IRN_Single_Trial_QC.csv')

YH28_1_X_R = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH28_240628_Kinematics_All.csv')
YH28_1_Y_R = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH28_240628_R_IRN_Single_Trial_QC.csv')

# 06/30
YH28_2_X_L = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH28_240630_Kinematics_All.csv')
YH28_2_Y_L = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH28_240630_L_IRN_Single_Trial_QC.csv')

YH28_2_X_R = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH28_240630_Kinematics_All.csv')
YH28_2_Y_R = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH28_240630_R_IRN_Single_Trial_QC.csv')

# 07/01
YH28_3_X_L = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH28_240701_Kinematics_All.csv')
YH28_3_Y_L = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH28_240701_L_IRN_Single_Trial_QC.csv')

YH28_3_X_R = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH28_240701_Kinematics_All.csv')
YH28_3_Y_R = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH28_240701_R_IRN_Single_Trial_QC.csv')

# 07/02
YH28_4_X_L = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH28_240702_Kinematics_All.csv')
YH28_4_Y_L = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH28_240702_L_IRN_Single_Trial_QC.csv')

YH28_4_X_R = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH28_240702_Kinematics_All.csv')
YH28_4_Y_R = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH28_240702_R_IRN_Single_Trial_QC.csv')




# Load Data - YH31

# 08/03
YH31_1_X_L = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH31_240803_Kinematics_All.csv')
YH31_1_Y_L = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH31_240803_L_IRN_Single_Trial_QC.csv')

YH31_1_X_R = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH31_240803_Kinematics_All.csv')
YH31_1_Y_R = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH31_240803_R_IRN_Single_Trial_QC.csv')

# 08/04
YH31_2_X_L = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH31_240804_Kinematics_All.csv')
YH31_2_Y_L = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH31_240804_L_IRN_Single_Trial_QC.csv')

YH31_2_X_R = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH31_240804_Kinematics_All.csv')
YH31_2_Y_R = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH31_240804_R_IRN_Single_Trial_QC.csv')

# 08/05
YH31_3_X_L = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH31_240805_Kinematics_All.csv')
YH31_3_Y_L = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH31_240805_L_IRN_Single_Trial_QC.csv')

YH31_3_X_R = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH31_240805_Kinematics_All.csv')
YH31_3_Y_R = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH31_240805_R_IRN_Single_Trial_QC.csv')

# 08/06
YH31_4_X_L = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH31_240806_Kinematics_All.csv')
YH31_4_Y_L = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH31_240806_L_IRN_Single_Trial_QC.csv')

YH31_4_X_R = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH31_240806_Kinematics_All.csv')
YH31_4_Y_R = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH31_240806_R_IRN_Single_Trial_QC.csv')

# 08/07
YH31_5_X_L = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH31_240807_Kinematics_All.csv')
YH31_5_Y_L = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH31_240807_L_IRN_Single_Trial_QC.csv')

YH31_5_X_R = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH31_240807_Kinematics_All.csv')
YH31_5_Y_R = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH31_240807_R_IRN_Single_Trial_QC.csv')




# Load Data - YH32

# 08/15
YH32_1_X_L = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH32_240815_Kinematics_All.csv')
YH32_1_Y_L = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH32_240815_L_IRN_Single_Trial_QC.csv')

# 08/22
YH32_2_X_R = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH32_240822_Kinematics_All.csv')
YH32_2_Y_R = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH32_240822_R_IRN_Single_Trial_QC.csv')

# 08/23
YH32_3_X_R = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH32_240823_Kinematics_All.csv')
YH32_3_Y_R = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH32_240823_R_IRN_Single_Trial_QC.csv')



# Load Data - YH38

# 09/28
YH38_1_X_L = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH38_240928_Kinematics_All.csv')
YH38_1_Y_L = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH38_240928_L_IRN_Single_Trial.csv')

YH38_1_X_R = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH38_240928_Kinematics_All.csv')
YH38_1_Y_R = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH38_240928_R_IRN_Single_Trial.csv')

# 09/29
YH38_2_X_L = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH38_240929_Kinematics_All.csv')
YH38_2_Y_L = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH38_240929_L_IRN_Single_Trial.csv')

YH38_2_X_R = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH38_240929_Kinematics_All.csv')
YH38_2_Y_R = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH38_240929_R_IRN_Single_Trial.csv')

# 09/30
YH38_3_X_L = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH38_240930_Kinematics_All.csv')
YH38_3_Y_L = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH38_240930_L_IRN_Single_Trial.csv')

YH38_3_X_R = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH38_240930_Kinematics_All.csv')
YH38_3_Y_R = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH38_240930_R_IRN_Single_Trial.csv')

# 10/01
YH38_4_X_L = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH38_241001_Kinematics_All.csv')
YH38_4_Y_L = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH38_241001_L_IRN_Single_Trial.csv')

YH38_4_X_R = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH38_241001_Kinematics_All.csv')
YH38_4_Y_R = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH38_241001_R_IRN_Single_Trial.csv')

# 10/02
YH38_5_X_L = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH38_241002_Kinematics_All.csv')
YH38_5_Y_L = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH38_241002_L_IRN_Single_Trial.csv')

YH38_5_X_R = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH38_241002_Kinematics_All.csv')
YH38_5_Y_R = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH38_241002_R_IRN_Single_Trial.csv')



# Load Data - YH39

# 10/01
YH39_1_X_L = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH39_241001_Kinematics_All.csv')
YH39_1_Y_L = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH39_241001_L_IRN_Single_Trial.csv')

YH39_1_X_R = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH39_241001_Kinematics_All.csv')
YH39_1_Y_R = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH39_241001_R_IRN_Single_Trial.csv')

# 10/02
YH39_2_X_L = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH39_241002_Kinematics_All.csv')
YH39_2_Y_L = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH39_241002_L_IRN_Single_Trial.csv')

YH39_2_X_R = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH39_241002_Kinematics_All.csv')
YH39_2_Y_R = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH39_241002_R_IRN_Single_Trial.csv')

# 10/03
YH39_3_X_L = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH39_241003_Kinematics_All.csv')
YH39_3_Y_L = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH39_241003_L_IRN_Single_Trial.csv')

YH39_3_X_R = pd.read_csv('/projectnb/economogrp/OE/Kinematics Data/YH39_241003_Kinematics_All.csv')
YH39_3_Y_R = pd.read_csv('/projectnb/economogrp/OE/Neural Data/YH39_241003_R_IRN_Single_Trial.csv')



# Data Preprocessing 
# PCA on Neural Data
# Z-scoring Kinematic Data
from scipy.stats import zscore 

def preprocess_data(X, Y, trial_len, session_id):

    
    # Drop Trial Type Column 
    if 'Trial Type' in X.columns: 
        X = X.drop(columns = ['Trial Type'])
    if 'Trial Type' in Y.columns: 
        Y = Y.drop(columns = ['Trial Type'])
    
    X = X.to_numpy()
    Y = Y.to_numpy()
    
    # Z - scoring Kinematic Features (X)
    # X = X[:,:-1]
    X = zscore(X, axis = 0)
    
    # Perform SVD on the activity data to get principal components
    A = (Y - np.mean(Y, axis=0)).T
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    
    # Eigenvalues
    var_exp_pc = [(S[i]**2) / np.sum(S**2) for i in range(len(S))]
    
    # Perform Dimensionality Reduction
    S = np.diag(S)
    # Y_pc = V[:, :PCs] @ S[:PCs, :PCs]
    Y_pc = V @ S
    

    # Session_ID array
    ID_array = session_id * np.ones((X.shape[0], 1))

    X_ID = np.concatenate((X, ID_array), axis = 1)
    Y_ID = np.concatenate((Y_pc, ID_array), axis = 1)

    X_reshaped = X_ID.reshape((X.shape[0] // trial_len, trial_len, X.shape[1] + 1))
    Y_reshaped = Y_ID.reshape((Y.shape[0] // trial_len, trial_len, Y.shape[1] + 1)) 


    return X_reshaped, Y_reshaped, var_exp_pc




# Generate New X and Y 
trial_len = 350

# --------------------------------------------------------- YH16 -----------------------------------------------------------
YH16_1_X_L_reshaped, YH16_1_Y_L_reshaped, YH16_var_exp_1_L = preprocess_data(YH16_1_X_L, YH16_1_Y_L, trial_len, 1)
YH16_1_X_R_reshaped, YH16_1_Y_R_reshaped, YH16_var_exp_1_R = preprocess_data(YH16_1_X_R, YH16_1_Y_R, trial_len, 2)

YH16_2_X_L_reshaped, YH16_2_Y_L_reshaped, YH16_var_exp_2_L = preprocess_data(YH16_2_X_L, YH16_2_Y_L, trial_len, 3)
YH16_2_X_R_reshaped, YH16_2_Y_R_reshaped, YH16_var_exp_2_R = preprocess_data(YH16_2_X_R, YH16_2_Y_R, trial_len, 4)

YH16_3_X_L_reshaped, YH16_3_Y_L_reshaped, YH16_var_exp_3_L = preprocess_data(YH16_3_X_L, YH16_3_Y_L, trial_len, 5)
YH16_3_X_R_reshaped, YH16_3_Y_R_reshaped, YH16_var_exp_3_R = preprocess_data(YH16_3_X_R, YH16_3_Y_R, trial_len, 6)

# --------------------------------------------------------- YH28 -----------------------------------------------------------
YH28_1_X_L_reshaped, YH28_1_Y_L_reshaped, YH28_var_exp_1_L = preprocess_data(YH28_1_X_L, YH28_1_Y_L, trial_len, 7)
YH28_1_X_R_reshaped, YH28_1_Y_R_reshaped, YH28_var_exp_1_R = preprocess_data(YH28_1_X_R, YH28_1_Y_R, trial_len, 8)

YH28_2_X_L_reshaped, YH28_2_Y_L_reshaped, YH28_var_exp_2_L = preprocess_data(YH28_2_X_L, YH28_2_Y_L, trial_len, 9)
YH28_2_X_R_reshaped, YH28_2_Y_R_reshaped, YH28_var_exp_2_R = preprocess_data(YH28_2_X_R, YH28_2_Y_R, trial_len, 10)

YH28_3_X_L_reshaped, YH28_3_Y_L_reshaped, YH28_var_exp_3_L = preprocess_data(YH28_3_X_L, YH28_3_Y_L, trial_len, 11)
YH28_3_X_R_reshaped, YH28_3_Y_R_reshaped, YH28_var_exp_3_R = preprocess_data(YH28_3_X_R, YH28_3_Y_R, trial_len, 12)

YH28_4_X_L_reshaped, YH28_4_Y_L_reshaped, YH28_var_exp_4_L = preprocess_data(YH28_4_X_L, YH28_4_Y_L, trial_len, 13)
YH28_4_X_R_reshaped, YH28_4_Y_R_reshaped, YH28_var_exp_4_R = preprocess_data(YH28_4_X_R, YH28_4_Y_R, trial_len, 14)


# --------------------------------------------------------- YH31 -----------------------------------------------------------
YH31_1_X_L_reshaped, YH31_1_Y_L_reshaped, YH31_var_exp_1_L = preprocess_data(YH31_1_X_L, YH31_1_Y_L, trial_len, 15)
YH31_1_X_R_reshaped, YH31_1_Y_R_reshaped, YH31_var_exp_1_R = preprocess_data(YH31_1_X_R, YH31_1_Y_R, trial_len, 16)

YH31_2_X_L_reshaped, YH31_2_Y_L_reshaped, YH31_var_exp_2_L = preprocess_data(YH31_2_X_L, YH31_2_Y_L, trial_len, 17)
YH31_2_X_R_reshaped, YH31_2_Y_R_reshaped, YH31_var_exp_2_R = preprocess_data(YH31_2_X_R, YH31_2_Y_R, trial_len, 18)

YH31_3_X_L_reshaped, YH31_3_Y_L_reshaped, YH31_var_exp_3_L = preprocess_data(YH31_3_X_L, YH31_3_Y_L, trial_len, 19)
YH31_3_X_R_reshaped, YH31_3_Y_R_reshaped, YH31_var_exp_3_R = preprocess_data(YH31_3_X_R, YH31_3_Y_R, trial_len, 20)

YH31_4_X_L_reshaped, YH31_4_Y_L_reshaped, YH31_var_exp_4_L = preprocess_data(YH31_4_X_L, YH31_4_Y_L, trial_len, 21)
YH31_4_X_R_reshaped, YH31_4_Y_R_reshaped, YH31_var_exp_4_R = preprocess_data(YH31_4_X_R, YH31_4_Y_R, trial_len, 22)

YH31_5_X_L_reshaped, YH31_5_Y_L_reshaped, YH31_var_exp_5_L = preprocess_data(YH31_5_X_L, YH31_5_Y_L, trial_len, 23)
YH31_5_X_R_reshaped, YH31_5_Y_R_reshaped, YH31_var_exp_5_R = preprocess_data(YH31_5_X_R, YH31_5_Y_R, trial_len, 24)

# --------------------------------------------------------- YH32 -----------------------------------------------------------
YH32_1_X_L_reshaped, YH32_1_Y_L_reshaped, YH32_var_exp_1_L = preprocess_data(YH32_1_X_L, YH32_1_Y_L, trial_len, 25)

YH32_2_X_R_reshaped, YH32_2_Y_R_reshaped, YH32_var_exp_2_R = preprocess_data(YH32_2_X_R, YH32_2_Y_R, trial_len, 26)

YH32_3_X_R_reshaped, YH32_3_Y_R_reshaped, YH32_var_exp_3_R = preprocess_data(YH32_3_X_R, YH32_3_Y_R, trial_len, 27)

# --------------------------------------------------------- YH32 -----------------------------------------------------------
YH38_1_X_L_reshaped, YH38_1_Y_L_reshaped, YH38_var_exp_1_L = preprocess_data(YH38_1_X_L, YH38_1_Y_L, trial_len, 28)
YH38_1_X_R_reshaped, YH38_1_Y_R_reshaped, YH38_var_exp_1_R = preprocess_data(YH38_1_X_R, YH38_1_Y_R, trial_len, 29)

YH38_2_X_L_reshaped, YH38_2_Y_L_reshaped, YH38_var_exp_2_L = preprocess_data(YH38_2_X_L, YH38_2_Y_L, trial_len, 30)
YH38_2_X_R_reshaped, YH38_2_Y_R_reshaped, YH38_var_exp_2_R = preprocess_data(YH38_2_X_R, YH38_2_Y_R, trial_len, 31)

YH38_3_X_L_reshaped, YH38_3_Y_L_reshaped, YH38_var_exp_3_L = preprocess_data(YH38_3_X_L, YH38_3_Y_L, trial_len, 32)
YH38_3_X_R_reshaped, YH38_3_Y_R_reshaped, YH38_var_exp_3_R = preprocess_data(YH38_3_X_R, YH38_3_Y_R, trial_len, 33)

YH38_4_X_L_reshaped, YH38_4_Y_L_reshaped, YH38_var_exp_4_L = preprocess_data(YH38_4_X_L, YH38_4_Y_L, trial_len, 34)
YH38_4_X_R_reshaped, YH38_4_Y_R_reshaped, YH38_var_exp_4_R = preprocess_data(YH38_4_X_R, YH38_4_Y_R, trial_len, 35)

YH38_5_X_L_reshaped, YH38_5_Y_L_reshaped, YH38_var_exp_5_L = preprocess_data(YH38_5_X_L, YH38_5_Y_L, trial_len, 36)
YH38_5_X_R_reshaped, YH38_5_Y_R_reshaped, YH38_var_exp_5_R = preprocess_data(YH38_5_X_R, YH38_5_Y_R, trial_len, 37)

# --------------------------------------------------------- YH39 -----------------------------------------------------------
YH39_1_X_L_reshaped, YH39_1_Y_L_reshaped, YH39_var_exp_1_L = preprocess_data(YH39_1_X_L, YH39_1_Y_L, trial_len, 38)
YH39_1_X_R_reshaped, YH39_1_Y_R_reshaped, YH39_var_exp_1_R = preprocess_data(YH39_1_X_R, YH39_1_Y_R, trial_len, 39)

YH39_2_X_L_reshaped, YH39_2_Y_L_reshaped, YH39_var_exp_2_L = preprocess_data(YH39_2_X_L, YH39_2_Y_L, trial_len, 40)
YH39_2_X_R_reshaped, YH39_2_Y_R_reshaped, YH39_var_exp_2_R = preprocess_data(YH39_2_X_R, YH39_2_Y_R, trial_len, 41)

YH39_3_X_L_reshaped, YH39_3_Y_L_reshaped, YH39_var_exp_3_L = preprocess_data(YH39_3_X_L, YH39_3_Y_L, trial_len, 42)
YH39_3_X_R_reshaped, YH39_3_Y_R_reshaped, YH39_var_exp_3_R = preprocess_data(YH39_3_X_R, YH39_3_Y_R, trial_len, 43)

# Set X and Y for Session CV
X = [YH16_1_X_L_reshaped, YH16_1_X_R_reshaped, YH16_2_X_L_reshaped, YH16_2_X_R_reshaped,
           YH16_3_X_L_reshaped, YH16_3_X_R_reshaped,
           YH28_1_X_L_reshaped, YH28_1_X_R_reshaped, YH28_2_X_L_reshaped, YH28_2_X_R_reshaped,
           YH28_3_X_L_reshaped, YH28_3_X_R_reshaped, YH28_4_X_L_reshaped, YH28_4_X_R_reshaped, 
           YH31_1_X_L_reshaped, YH31_1_X_R_reshaped, YH31_2_X_L_reshaped, YH31_2_X_R_reshaped,
           YH31_3_X_L_reshaped, YH31_3_X_R_reshaped, YH31_4_X_L_reshaped, YH31_4_X_R_reshaped,
           YH31_5_X_L_reshaped, YH31_5_X_R_reshaped,
           YH32_1_X_L_reshaped, YH32_2_X_R_reshaped, YH32_3_X_R_reshaped,
           YH38_1_X_L_reshaped, YH38_1_X_R_reshaped, YH38_2_X_L_reshaped, YH38_2_X_R_reshaped, 
           YH38_3_X_L_reshaped, YH38_3_X_R_reshaped, YH38_4_X_L_reshaped, YH38_4_X_R_reshaped,
           YH38_5_X_L_reshaped, YH38_5_X_R_reshaped, 
           YH39_1_X_L_reshaped, YH39_1_X_R_reshaped, YH39_2_X_L_reshaped, YH39_2_X_R_reshaped,
           YH39_3_X_L_reshaped, YH39_3_X_R_reshaped]

Y = [YH16_1_Y_L_reshaped, YH16_1_Y_R_reshaped, YH16_2_Y_L_reshaped, YH16_2_Y_R_reshaped,
           YH16_3_Y_L_reshaped, YH16_3_Y_R_reshaped,
           YH28_1_Y_L_reshaped, YH28_1_Y_R_reshaped, YH28_2_Y_L_reshaped, YH28_2_Y_R_reshaped,
           YH28_3_Y_L_reshaped, YH28_3_Y_R_reshaped, YH28_4_Y_L_reshaped, YH28_4_Y_R_reshaped, 
           YH31_1_Y_L_reshaped, YH31_1_Y_R_reshaped, YH31_2_Y_L_reshaped, YH31_2_Y_R_reshaped,
           YH31_3_Y_L_reshaped, YH31_3_Y_R_reshaped, YH31_4_Y_L_reshaped, YH31_4_Y_R_reshaped,
           YH31_5_Y_L_reshaped, YH31_5_Y_R_reshaped,
           YH32_1_Y_L_reshaped, YH32_2_Y_R_reshaped, YH32_3_Y_R_reshaped,
           YH38_1_Y_L_reshaped, YH38_1_Y_R_reshaped, YH38_2_Y_L_reshaped, YH38_2_Y_R_reshaped, 
           YH38_3_Y_L_reshaped, YH38_3_Y_R_reshaped, YH38_4_Y_L_reshaped, YH38_4_Y_R_reshaped,
           YH38_5_Y_L_reshaped, YH38_5_Y_R_reshaped, 
           YH39_1_Y_L_reshaped, YH39_1_Y_R_reshaped, YH39_2_Y_L_reshaped, YH39_2_Y_R_reshaped,
           YH39_3_Y_L_reshaped, YH39_3_Y_R_reshaped]

# Test and Train split for Different Sessions
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split

# Function for creating Data Loaders
def create_dataloader(X, Y, batch_size=20):
    data = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return dataloader

# Create Loaders for YH16 
loader_YH16_1_L = create_dataloader(YH16_1_X_L_reshaped, YH16_1_Y_L_reshaped)
loader_YH16_1_R = create_dataloader(YH16_1_X_R_reshaped, YH16_1_Y_R_reshaped)
loader_YH16_2_L = create_dataloader(YH16_2_X_L_reshaped, YH16_2_Y_L_reshaped)
loader_YH16_2_R = create_dataloader(YH16_2_X_R_reshaped, YH16_2_Y_R_reshaped)
loader_YH16_3_L = create_dataloader(YH16_3_X_L_reshaped, YH16_3_Y_L_reshaped)
loader_YH16_3_R = create_dataloader(YH16_3_X_R_reshaped, YH16_3_Y_R_reshaped)


# Create Loaders for YH28
loader_YH28_1_L = create_dataloader(YH28_1_X_L_reshaped, YH28_1_Y_L_reshaped)
loader_YH28_1_R = create_dataloader(YH28_1_X_R_reshaped, YH28_1_Y_R_reshaped)
loader_YH28_2_L = create_dataloader(YH28_2_X_L_reshaped, YH28_2_Y_L_reshaped)
loader_YH28_2_R = create_dataloader(YH28_2_X_R_reshaped, YH28_2_Y_R_reshaped)
loader_YH28_3_L = create_dataloader(YH28_3_X_L_reshaped, YH28_3_Y_L_reshaped)
loader_YH28_3_R = create_dataloader(YH28_3_X_R_reshaped, YH28_3_Y_R_reshaped)
loader_YH28_4_L = create_dataloader(YH28_4_X_L_reshaped, YH28_4_Y_L_reshaped)
loader_YH28_4_R = create_dataloader(YH28_4_X_R_reshaped, YH28_4_Y_R_reshaped)



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


# Create Loaders for YH32 
loader_YH32_1_L = create_dataloader(YH32_1_X_L_reshaped, YH32_1_Y_L_reshaped)
loader_YH32_2_R = create_dataloader(YH32_2_X_R_reshaped, YH32_2_Y_R_reshaped)
loader_YH32_3_R = create_dataloader(YH32_3_X_R_reshaped, YH32_3_Y_R_reshaped)


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


# Create Loaders for YH39
loader_YH39_1_L = create_dataloader(YH39_1_X_L_reshaped, YH39_1_Y_L_reshaped)
loader_YH39_1_R = create_dataloader(YH39_1_X_R_reshaped, YH39_1_Y_R_reshaped)
loader_YH39_2_L = create_dataloader(YH39_2_X_L_reshaped, YH39_2_Y_L_reshaped)
loader_YH39_2_R = create_dataloader(YH39_2_X_R_reshaped, YH39_2_Y_R_reshaped)
loader_YH39_3_L = create_dataloader(YH39_3_X_L_reshaped, YH39_3_Y_L_reshaped)
loader_YH39_3_R = create_dataloader(YH39_3_X_R_reshaped, YH39_3_Y_R_reshaped)

# Print Variance Explained for each Animal
YH16_var_exp = [YH16_var_exp_1_L, YH16_var_exp_1_R, YH16_var_exp_2_L, YH16_var_exp_2_R, YH16_var_exp_3_L, YH16_var_exp_3_R]
YH28_var_exp = [YH28_var_exp_1_L, YH28_var_exp_1_R, YH28_var_exp_2_L, YH28_var_exp_2_R, YH28_var_exp_3_L, YH28_var_exp_3_R, YH28_var_exp_4_L, YH28_var_exp_4_R]
YH31_var_exp = [YH31_var_exp_1_L, YH31_var_exp_1_R, YH31_var_exp_2_L, YH31_var_exp_2_R, YH31_var_exp_3_L, YH31_var_exp_3_R, YH31_var_exp_4_L, YH31_var_exp_4_R, YH31_var_exp_5_L, YH31_var_exp_5_R]
YH32_var_exp = [YH32_var_exp_1_L, YH32_var_exp_2_R, YH32_var_exp_3_R]
YH38_var_exp = [YH38_var_exp_1_L, YH38_var_exp_1_R, YH38_var_exp_2_L, YH38_var_exp_2_R, YH38_var_exp_3_L, YH38_var_exp_3_R, YH38_var_exp_4_L, YH38_var_exp_4_R, YH38_var_exp_5_L, YH38_var_exp_5_R]
YH39_var_exp = [YH39_var_exp_1_L, YH39_var_exp_1_R, YH39_var_exp_2_L, YH39_var_exp_2_R, YH39_var_exp_3_L, YH39_var_exp_3_R]

Actual_Var_Exp = [YH16_var_exp, YH28_var_exp, YH31_var_exp, YH32_var_exp, YH38_var_exp, YH39_var_exp]

# Extract Batches Function
def extract_batches(loader):
    batches = []
    for batch in loader: 
        batches.append(batch)
    return batches