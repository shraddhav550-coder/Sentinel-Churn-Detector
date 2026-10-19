
import pandas as pd
from pathlib import Path

# 1. Load the dataset
file_path = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = pd.read_csv(file_path)

print("INITIAL DATASET SHAPE:", df.shape)

# 2. Remove duplicate records
duplicates = df.duplicated().sum()
print("\nDuplicate rows found:", duplicates)

df = df.drop_duplicates()

# 3. Check missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# 4. Clean TotalCharges
# Some blank values may be stored as spaces
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

print("\nMissing TotalCharges after conversion:",
      df["TotalCharges"].isnull().sum())

# 5. Check the target column
print("\nChurn values:")
print(df["Churn"].value_counts(dropna=False))

# 6. Remove rows with missing essential values
df = df.dropna(subset=["Churn", "TotalCharges"])

# 7. Remove customer IDs from model features later
# IDs identify records; they are not useful behavioral predictors.

print("\nCLEANED DATASET SHAPE:", df.shape)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nData cleaning completed successfully!")
