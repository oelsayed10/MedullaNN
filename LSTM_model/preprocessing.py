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