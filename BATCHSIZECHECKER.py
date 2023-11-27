import pandas as pd
import numpy as np
import torch

# Load data from a CSV file
csv_file_path = '/Users/josiahmccarthy/Documents/SCHOOL/SNR CAP DESIGN/Database_Senior_Project/Split_data/Gyro/Training.csv'  #Replace with the path to your CSV file
data = pd.read_csv(csv_file_path)

# Assuming your data contains numeric values in a format that can be directly converted to a tensor
# You may need to preprocess the data according to your specific dataset
# For example, normalizing or scaling values

# Convert data to a numpy array
data_array = data.to_numpy()

# Define the desired dimensions
batch_size = 1  # Change this value to match your batch size
channels = 1    # Change this value to match the number of channels
height = data_array.shape[0]  # Assuming data samples correspond to height
width = data_array.shape[1]   # Assuming data features correspond to width

# Reshape the data into a 4D tensor
data_tensor = torch.tensor(data_array, dtype=torch.float32)
data_tensor = data_tensor.view(batch_size, channels, height, width)

# Now, data_tensor is a 4D tensor with dimensions (batch_size, channels, height, width)