import pandas as pd

# Function to display rows with NaN values in specific columns or entire dataset
def display_nan_data(df):
    # Display rows with NaN values in specific columns
    nan_rows = df[df.isna().any(axis=1)]
    
    if nan_rows.empty:
        print("Tidak ada nilai NaN di dataset.")
    else:
        print("Baris dengan nilai NaN:")
        print(nan_rows)

# Assume df is your DataFrame
file_path = 'path_to_your_file.xlsx'
df = pd.read_excel(file_path, engine='openpyxl')

# Display NaN data
display_nan_data(df)
