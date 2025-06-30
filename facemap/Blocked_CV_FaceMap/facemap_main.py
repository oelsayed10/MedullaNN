# FaceMap Model Main - Omar

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from facemap_preprocessing import X, Y, Actual_Var_Exp, create_dataloader, extract_batches
from facemap_train import train_and_test_model
from facemap_model import KeypointsNetwork
import sys



# Connect to Device: 
# Device Object - CPU and GPU 
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device", device)

# Set Model Parameters 
out_list = [80, 55, 96, 71, 62, 47, 72, 60, 101,
            65, 45, 51, 54, 51, 61, 52, 77, 61,
            86, 65, 36, 52, 59, 37, 48, 59, 44,
            59, 71, 89, 92, 95, 65, 100, 76, 63,
            82, 43, 54, 49, 47, 103, 67]

sessions_list = np.arange(1, 44, 1)

# Set Training Parameters
Loss_fcn = nn.MSELoss()
num_epochs = int(sys.argv[2])
Animal = int(sys.argv[1])
FOLD_NO = int(sys.argv[3])

# Model Training
PERFORMANCES = train_and_test_model(X, Y, KeypointsNetwork,
                                    out_list, sessions_list, Loss_fcn, extract_batches, 
                                    create_dataloader, device, num_epochs,
                                    Animal, FOLD_NO)


# Variance Explained by Model

Var_exp_actual = Actual_Var_Exp[Animal-1]

Var_exp_MODEL = [np.dot(PERFORMANCES[i], Var_exp_actual[i]) for i in range(len(PERFORMANCES))]

# Save as CSV 
df = pd.DataFrame(Var_exp_MODEL)
df.to_csv(f'Variance_exp_BlockedCV_{Animal}_FOLD{FOLD_NO}_FaceMap.csv', index = False)

print('List Saved Successfully')





