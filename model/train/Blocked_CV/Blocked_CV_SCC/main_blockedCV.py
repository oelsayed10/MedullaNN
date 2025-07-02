# Main File for Blocked CV 

# Import Libraries 
import numpy as np
import pandas as pd
import torch 
from sklearn.metrics import r2_score
import torch.optim as optim
import torch.nn as nn
from model_blocked import LSTM_model
from pre_processing_BlockedCV import X, Y, Actual_Var_Exp, create_dataloader, extract_batches
from train_model_BlockedCV import train_and_test_model
import sys


Animal = int(sys.argv[1])
FOLD_NO = int(sys.argv[3])

# Connect to Device: 
# Device Object - CPU and GPU 
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device", device)

# Set Model Parameters 
input_dim = 55  # Input dimension to LSTM
hidden_dim1 = 128  # Hidden dimension of LSTM
num_layers = 3  # LSTM Layers

# Set Training Parameters
Loss_fcn = nn.MSELoss()
num_epochs = int(sys.argv[2])

PERFORMANCES = train_and_test_model(X, Y, LSTM_model, input_dim, hidden_dim1, num_layers, Loss_fcn, extract_batches, create_dataloader, device, num_epochs, Animal, FOLD_NO)


# Variance Explained by Model

Var_exp_actual = Actual_Var_Exp[Animal-1]

Var_exp_MODEL = [np.dot(PERFORMANCES[i], Var_exp_actual[i]) for i in range(len(PERFORMANCES))]

# Save as CSV 
df = pd.DataFrame(Var_exp_MODEL)
df.to_csv(f'Variance_exp_BlockedCV_{Animal}_FOLD{FOLD_NO}.csv', index = False)

print('List Saved Successfully')
