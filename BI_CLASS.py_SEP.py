# # EEG

# import pandas as pd

# # Load the CSV file
# file_path = '/Users/josiahmccarthy/Documents/SCHOOL/SNR CAP DESIGN/Database_Senior_Project/Split_data/EEG/Training.csv'  # Replace with the actual file path
# df = pd.read_csv(file_path)

# # Modify the 'Label' column values
# df['Label'] = df['Label'].replace({1: 0, 3: 1})

# # Specify the full output file path, including the filename
# output_file_path = '/Users/josiahmccarthy/Documents/SCHOOL/SNR CAP DESIGN/Database_Senior_Project/Final_Split_data[binaryclassification].csv'  # Replace with the desired output file path
# df.to_csv(output_file_path, index=False)

# print("EEG File saved successfully.")


# #FINAL_EEG_TEST[binary]

# GYRO

import pandas as pd

# Load the CSV file
file_path = '/Users/josiahmccarthy/Documents/SCHOOL/SNR CAP DESIGN/Database_Senior_Project/Split_data/Gyro/Training.csv'  # Replace with the actual file path
df = pd.read_csv(file_path)

# Modify the 'Label' column values
df['Label'] = df['Label'].replace({4: 0, 2: 1})

# Specify the full output file path, including the filename
output_file_path = '/Users/josiahmccarthy/Documents/SCHOOL/SNR CAP DESIGN/Database_Senior_Project/Final_Split_data[binaryclassification].csv'  # Replace with the desired output file path
df.to_csv(output_file_path, index=False)

print("GYRO File saved successfully.")

# #FINAL_GYRO_TEST[binary]