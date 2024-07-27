import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def handle_missing_values(df, strategy='mean', columns=None):
    """Handle missing values in the DataFrame."""
    if columns is None:
        columns = df.select_dtypes(include=['float64', 'int64']).columns

    for column in columns:
        if strategy == 'mean':
            df[column] = df[column].fillna(df[column].mean())
        elif strategy == 'median':
            df[column] = df[column].fillna(df[column].median())
        elif strategy == 'mode':
            df[column] = df[column].fillna(df[column].mode().iloc[0])
        elif strategy == 'drop':
            df = df.dropna(subset=[column])

    return df

def remove_duplicates(df):
    """Remove duplicate rows from the DataFrame."""
    return df.drop_duplicates()

def normalize_data(df, columns=None):
    """Normalize numeric columns in the DataFrame."""
    if columns is None:
        columns = df.select_dtypes(include=['float64', 'int64']).columns

    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])

    return df

def transform_data_types(df, column_type_mapping):
    """Transform data types of specific columns in the DataFrame."""
    for column, dtype in column_type_mapping.items():
        if column in df.columns:
            try:
                df[column] = df[column].astype(dtype)
            except ValueError:
                print(f"Warning: Unable to convert column '{column}' to {dtype}")
        else:
            print(f"Warning: Column '{column}' not found in DataFrame.")
    return df

def save_data(df, file_path):
    """Save the cleaned DataFrame to a CSV file."""
    df.to_csv(file_path, index=False)

# Input and output file paths
file_path = r"C:\Users\hp\Downloads\Literacy Rate.csv"
output_file_path = r"C:\Users\hp\Downloads\Cleaned_Literacy_Rate.csv"

# Load the data
df = load_data(file_path)

# Handle missing values
df = handle_missing_values(df, strategy='mean')

# Remove duplicates
df = remove_duplicates(df)

# Normalize data
df = normalize_data(df)

# Save the cleaned data
save_data(df, output_file_path)

print("Data cleaning and preprocessing complete.")
