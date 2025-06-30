# Preprocessing

# Import Libraries 
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd
import torch 
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import random

def preprocess_features(X, trial_len):

    # Drop Trial Type Column if exists 
    if 'Trial Type' in X.columns: 
        X = X.drop(columns = ['Trial Type'])
    
    X = X.to_numpy()
    
    # Z - scoring Kinematic Features (X)
    X = zscore(X, axis = 0)

    # Reshape (Trials, Time Points, Features + ID)
    X_reshaped = X.reshape((X.shape[0] // trial_len, trial_len, X.shape[1])).astype(np.float32)
    X_reshaped = torch.from_numpy(X_reshaped)

    return X_reshaped


# Set Model Parameters 
out_list = [80, 55, 96, 71, 62, 47, 72, 60, 101,
            65, 45, 51, 54, 51, 61, 52, 77, 61,
            86, 65, 36, 52, 59, 37, 48, 59, 44,
            59, 71, 89, 92, 95, 65, 100, 76, 63,
            82, 43, 54, 49, 47, 103, 67]

sessions_list = np.arange(1, 44, 1)