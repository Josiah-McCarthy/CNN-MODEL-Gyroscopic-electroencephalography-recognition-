import pandas as pd

# Load your CSV file
file_path = '/Users/josiahmccarthy/Documents/SCHOOL/SNR CAP DESIGN/Database_Senior_Project/FINAL_SPLIT_DATA_2/EEG_TEST_2.csv'
data = pd.read_csv(file_path)

# Check for values in the "Label" column that are not 0 or 1
invalid_labels = data.loc[~data['Label'].isin([0, 1]), 'Label']

# Print the invalid labels, if any
if not invalid_labels.empty:
    print("Invalid labels found in 'Label' column:")
    print(invalid_labels)
else:
    print("No invalid labels found in 'Label' column.")
