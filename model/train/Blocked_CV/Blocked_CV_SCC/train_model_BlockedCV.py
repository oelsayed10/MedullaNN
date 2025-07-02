# Train and Test Model - Blocked CV 

# Import Libraries 
import numpy as np
import pandas as pd
import torch 
from sklearn.metrics import r2_score
import torch.optim as optim
import time
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
import random

# Suppress the specific warning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def train_and_test_model(X, Y, LSTM_model, input_dim, hidden_dim1, num_layers, Loss_fcn, extract_batches, create_dataloader, device, num_epochs, Animal, FOLD_NO):
    model = LSTM_model(input_dim, hidden_dim1, num_layers, Animal)
    model.to(device)

    Loaders = []
    for i in range(len(X)):
        loader = create_dataloader(X[i], Y[i])
        Loaders.append(loader)


    if Animal == 0: 
        train_loaders = Loaders
        test_loaders = None
    elif Animal == 1: 
        train_loaders = Loaders[6:43]
        test_loaders = Loaders[0:6]
    elif Animal == 2:
        train_loaders = Loaders[0:6] + Loaders[14:43]
        test_loaders = Loaders[6:14]
    elif Animal == 3:
        train_loaders = Loaders[0:14] + Loaders[24:43]
        test_loaders = Loaders[14:24]
    elif Animal == 4: 
        train_loaders = Loaders[0:24] + Loaders[27:43]
        test_loaders = Loaders[24:27]
    elif Animal == 5: 
        train_loaders = Loaders[0:27] + Loaders[37:43]
        test_loaders = Loaders[27:37]
    elif Animal == 6: 
        train_loaders = Loaders[0:37]
        test_loaders = Loaders[37:43] 



    all_batches = []
    for loader in train_loaders: 
        all_batches.append(extract_batches(loader))

    combined_batches = [batch for session_batches in all_batches for batch in session_batches]
    random.shuffle(combined_batches)

    train_loader = DataLoader(combined_batches, batch_size=1, shuffle=False)

    if test_loaders is None: 
        print(f'Size of Train Loaders = {len(train_loaders)}')
    else: 
        print(f'Size of Train and Test Loaders = {len(train_loaders) + len(test_loaders)}')

    # Model Training - Cross Animal --- Train on three animals ---> Test on the fourth
    import time 

    start_time = time.time()

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

    best_r2 = -float('inf')
    best_loss = float('inf')

    if test_loaders is None:
        model_save_path = f'Model_Trained_on_All.pth'
    else: 
        model_save_path = f'Model_tested_on_{Animal}_FOLD{FOLD_NO}.pth'

    train_Rsquared = []
    print(f'Training Started')

    for epoch in range(num_epochs):
        model.train()
        train_r2 = 0.0

        for X_batch, Y_batch in train_loader:
            features = X_batch.to(device).squeeze(0)
            target = Y_batch.to(device).squeeze(0)
            optimizer.zero_grad()

            target = target[:, :, :-1]
            outputs = model(features)

            loss = Loss_fcn(outputs, target)
            loss.backward()
            optimizer.step()
            
            outputs_np = outputs.detach().cpu().numpy()
            target_np = target.cpu().numpy()
            r2_batch = r2_score(target_np.reshape(-1, target_np.shape[-1]), outputs_np.reshape(-1, outputs_np.shape[-1]), multioutput='uniform_average')
            train_r2 += r2_batch

        # pc_r2 /= len(train_loader.dataset)
        train_r2 /= len(train_loader.dataset)
        train_Rsquared.append(train_r2)
        
        if loss.item() < best_loss:
            best_loss = loss.item()

        if train_r2 > best_r2:
            best_r2 = train_r2
            torch.save(model.state_dict(), model_save_path)

        # if (epoch + 1) % 100 == 0:
        #     print(f'Epoch [{epoch+1}/{num_epochs}], R2: {train_r2:.4f}')
    
    print('BEST MSE LOSS: ', best_loss)
    print('BEST R2: ', best_r2)


    end_time = time.time()
    print(f'Training Done {(end_time - start_time) / 60 :.2f} Minutes')

    if test_loaders is None:
        print('Training Complemeted \n Model Saved')

    else:
        PERFORMANCES = []
        for test_loader in test_loaders:
            model.load_state_dict(torch.load(f'Model_tested_on_{Animal}_FOLD{FOLD_NO}.pth'))
            model.eval()

            ID = int(test_loader.dataset[9][0][0, -1])
            print(f'Test_Loader_ID:', ID)

            all_targets = test_loader.dataset[:][1][:, :, :-1]
            all_features = (test_loader.dataset[:][0][:, :, :]).to(device)

            with torch.no_grad():
                all_outputs = model(all_features)
                # print(all_outputs.shape)

            X_regression = all_outputs.cpu().numpy().reshape(-1, all_outputs.shape[-1])
            Y_regression = all_targets.cpu().numpy().reshape(-1, all_targets.shape[-1])

            BETA_OLS = np.linalg.lstsq(X_regression, Y_regression, rcond=None)
            # print(BETA_OLS[0].shape)

            Y_hat = X_regression @ BETA_OLS[0]
            r2_pc_reg = [r2_score(Y_regression[:, i], Y_hat[:, i]) for i in range(Y_regression.shape[1])]
            # Y_hat_reshaped = Y_hat.reshape(all_targets.shape[0], all_targets.shape[1], all_targets.shape[2])
            # print(Y_hat_reshaped.shape[0])

            # PERFORMANCES.append(Y_hat_reshaped.shape[0])
            PERFORMANCES.append(r2_pc_reg)
            print(f"Completed Evaluation for Loader {ID} - {test_loader}")
            print('')


        return PERFORMANCES

    



