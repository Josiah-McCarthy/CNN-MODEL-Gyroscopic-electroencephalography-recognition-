import pandas as pd
from sklearn.utils import shuffle

# Load the input CSV file
input_file = '/Users/josiahmccarthy/Documents/SCHOOL/SNR CAP DESIGN/Database_Senior_Project/Final_stage_data/Split_data/Training_data.csv'
data = pd.read_csv(input_file)

# Split the data into two separate dataframes
motg_data = data[['MOT.Q0', 'MOT.Q1', 'MOT.Q2', 'MOT.Q3' ,'Label']]
af_data = data[['EEG.AF3', 'EEG.AF4', 'Label']]

# Randomize the data
motg_data = shuffle(motg_data)
af_data = shuffle(af_data)

# Save the two dataframes to separate CSV files
motg_data.to_csv('GYRO_TRAIN.csv', index=False)
af_data.to_csv('EEG_TRAIN', index=False)
