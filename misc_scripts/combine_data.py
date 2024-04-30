import pandas as pd

# List of your train split files
splits = ['data/train_split_1.pkl', 'data/train_split_2.pkl', 'data/train_split_3.pkl', 'data/train_split_4.pkl']

# Load each file as a pandas DataFrame and append it to a list
dataframes = [pd.read_pickle(split) for split in splits]

# Concatenate all dataframes into one
combined_dataframe = pd.concat(dataframes, ignore_index=True)

# Save the combined dataframe to a new .pkl file
combined_dataframe.to_pickle('data/train_combined.pkl')

print("Combined dataset saved to 'data/train_combined.pkl'.")

