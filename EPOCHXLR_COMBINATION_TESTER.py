import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import time

class ImprovedGestureClassifier(nn.Module):
    def __init__(self, num_classes, input_channels):
        super(ImprovedGestureClassifier, self).__init__()
        self.input_channels = input_channels
        self.fc1 = nn.Linear(input_channels, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        return out

def train_and_evaluate(gyro_model, eeg_model, gyro_optimizer, eeg_optimizer, gyro_criterion, eeg_criterion, gyro_train_loader, eeg_train_loader, gyro_test_loader, eeg_test_loader, num_epochs_gyro, num_epochs_eeg, lr_gyro, lr_eeg):
    start_time = time.time()

    # Training loop for GYRO
    for epoch in range(num_epochs_gyro):
        gyro_model.train()
        for inputs, targets in gyro_train_loader:
            gyro_optimizer.zero_grad()
            outputs = gyro_model(inputs)
            loss = gyro_criterion(outputs, targets)
            loss.backward()
            gyro_optimizer.step()

    # Evaluation loop for GYRO
    gyro_model.eval()
    gyro_total_correct = 0
    gyro_total_samples = 0

    with torch.no_grad():
        for inputs, targets in gyro_test_loader:
            outputs = gyro_model(inputs)
            predicted_classes = torch.argmax(outputs, dim=1)
            gyro_total_correct += (predicted_classes == targets).sum().item()
            gyro_total_samples += targets.size(0)

    gyro_accuracy = gyro_total_correct / gyro_total_samples

    # Training loop for EEG
    for epoch in range(num_epochs_eeg):
        eeg_model.train()
        for inputs, targets in eeg_train_loader:
            eeg_optimizer.zero_grad()
            outputs = eeg_model(inputs)
            loss = eeg_criterion(outputs, targets)
            loss.backward()
            eeg_optimizer.step()

    # Evaluation loop for EEG
    eeg_model.eval()
    eeg_total_correct = 0
    eeg_total_samples = 0

    with torch.no_grad():
        for inputs, targets in eeg_test_loader:
            outputs = eeg_model(inputs)
            predicted_classes = torch.argmax(outputs, dim=1)
            eeg_total_correct += (predicted_classes == targets).sum().item()
            eeg_total_samples += targets.size(0)

    eeg_accuracy = eeg_total_correct / eeg_total_samples

    end_time = time.time()
    elapsed_time = end_time - start_time

    return gyro_accuracy, eeg_accuracy, elapsed_time

# Load GYRO training and testing data
# (Replace the file paths with your actual file paths)
gyro_train_data = pd.read_csv('/Users/josiahmccarthy/Documents/SCHOOL/SNR CAP DESIGN/Database_Senior_Project/Final_Split_data[binaryclassification]/FINAL_GYRO_TRAIN[binary].csv')
gyro_test_data = pd.read_csv('/Users/josiahmccarthy/Documents/SCHOOL/SNR CAP DESIGN/Database_Senior_Project/Final_Split_data[binaryclassification]/FINAL_GYRO_TEST[binary].csv')

X_gyro_train = gyro_train_data[['MOT.Q0', 'MOT.Q1']].values
Y_gyro_train = gyro_train_data['Label'].values
X_gyro_test = gyro_test_data[['MOT.Q0', 'MOT.Q1']].values
Y_gyro_test = gyro_test_data['Label'].values

X_gyro_train_tensor = torch.tensor(X_gyro_train, dtype=torch.float32)
Y_gyro_train_tensor = torch.tensor(Y_gyro_train, dtype=torch.long)
X_gyro_test_tensor = torch.tensor(X_gyro_test, dtype=torch.float32)
Y_gyro_test_tensor = torch.tensor(Y_gyro_test, dtype=torch.long)

gyro_train_data = TensorDataset(X_gyro_train_tensor, Y_gyro_train_tensor)
gyro_train_loader = DataLoader(gyro_train_data, batch_size=32, shuffle=True)
gyro_test_data = TensorDataset(X_gyro_test_tensor, Y_gyro_test_tensor)
gyro_test_loader = DataLoader(gyro_test_data, batch_size=32, shuffle=False)

# Load EEG training and testing data
# (Replace the file paths with your actual file paths)
eeg_train_data = pd.read_csv('/Users/josiahmccarthy/Documents/SCHOOL/SNR CAP DESIGN/Database_Senior_Project/Final_Split_data[binaryclassification]/FINAL_EEG_TRAIN[binary].csv')
eeg_test_data = pd.read_csv('/Users/josiahmccarthy/Documents/SCHOOL/SNR CAP DESIGN/Database_Senior_Project/Final_Split_data[binaryclassification]/FINAL_EEG_TEST[binary].csv')

X_eeg_train = eeg_train_data[['EEG.AF3', 'EEG.AF4']].values
Y_eeg_train = eeg_train_data['Label'].values
X_eeg_test = eeg_test_data[['EEG.AF3', 'EEG.AF4']].values
Y_eeg_test = eeg_test_data['Label'].values

X_eeg_train_tensor = torch.tensor(X_eeg_train, dtype=torch.float32)
Y_eeg_train_tensor = torch.tensor(Y_eeg_train, dtype=torch.long)
X_eeg_test_tensor = torch.tensor(X_eeg_test, dtype=torch.float32)
Y_eeg_test_tensor = torch.tensor(Y_eeg_test, dtype=torch.long)

eeg_train_data = TensorDataset(X_eeg_train_tensor, Y_eeg_train_tensor)
eeg_train_loader = DataLoader(eeg_train_data, batch_size=32, shuffle=True)
eeg_test_data = TensorDataset(X_eeg_test_tensor, Y_eeg_test_tensor)
eeg_test_loader = DataLoader(eeg_test_data, batch_size=32, shuffle=False)

# Define hyperparameter search ranges
lr_range = [0.1, 0.01, 0.001, 0.0001, 0.00001]
epoch_range = list(range(1, 61))

# Initialize variables to store the best configuration
best_gyro_accuracy = 0
best_eeg_accuracy = 0
best_gyro_lr = 0
best_eeg_lr = 0
best_gyro_epochs = 0
best_eeg_epochs = 0

# Initialize DataFrame to store results
results_df = pd.DataFrame(columns=['GYRO Testing Accuracy', 'EEG Testing Accuracy', 'Total Execution Time', 'Value of epochs for GYRO', 'Value of epochs for EEG', 'Value of lr for GYRO', 'Value of lr for EEG'])

# Grid search
start_search_time = time.time()  # Record start time of the grid search
for lr_gyro in lr_range:
    for lr_eeg in lr_range:
        for epochs_gyro in epoch_range:
            for epochs_eeg in epoch_range:
                # Initialize models and optimizers
                gyro_model = ImprovedGestureClassifier(num_classes=len(set(Y_gyro_train)), input_channels=2)
                eeg_model = ImprovedGestureClassifier(num_classes=len(set(Y_eeg_train)), input_channels=2)
                gyro_optimizer = optim.Adam(gyro_model.parameters(), lr=lr_gyro)
                eeg_optimizer = optim.Adam(eeg_model.parameters(), lr=lr_eeg)
                gyro_criterion = nn.CrossEntropyLoss()
                eeg_criterion = nn.CrossEntropyLoss()

                # Train and evaluate models
                gyro_accuracy, eeg_accuracy, elapsed_time = train_and_evaluate(gyro_model, eeg_model, gyro_optimizer, eeg_optimizer, gyro_criterion, eeg_criterion, gyro_train_loader, eeg_train_loader, gyro_test_loader, eeg_test_loader, epochs_gyro, epochs_eeg, lr_gyro, lr_eeg)

                # Update best configuration if a better one is found
                if gyro_accuracy > best_gyro_accuracy:
                    best_gyro_accuracy = gyro_accuracy
                    best_gyro_lr = lr_gyro
                    best_gyro_epochs = epochs_gyro

                if eeg_accuracy > best_eeg_accuracy:
                    best_eeg_accuracy = eeg_accuracy
                    best_eeg_lr = lr_eeg
                    best_eeg_epochs = epochs_eeg

                # Append results to DataFrame
                results_df = pd.concat([results_df, pd.DataFrame({
                 'GYRO Testing Accuracy': gyro_accuracy,
                  'EEG Testing Accuracy': eeg_accuracy,
                  'Total Execution Time': elapsed_time,
                 'Value of epochs for GYRO': epochs_gyro,
                    'Value of epochs for EEG': epochs_eeg,   
                   'Value of lr for GYRO': lr_gyro,
                   'Value of lr for EEG': lr_eeg
                }, index=[0])], ignore_index=True)

                # Print elapsed time every 10 seconds
                current_time = time.time()
                if current_time - start_search_time >= 10:
                    print(f"Elapsed Time: {current_time - start_search_time:.2f} seconds")
                    start_search_time = current_time


# Save results to CSV
results_df.to_csv('grid_search_results.csv', index=False)

# Print best configuration
print(f"Best GYRO Configuration - Accuracy: {best_gyro_accuracy * 100:.2f}%, LR: {best_gyro_lr}, Epochs: {best_gyro_epochs}")
print(f"Best EEG Configuration - Accuracy: {best_eeg_accuracy * 100:.2f}%, LR: {best_eeg_lr}, Epochs: {best_eeg_epochs}")
