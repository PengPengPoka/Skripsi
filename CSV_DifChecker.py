import pandas as pd
import numpy as np

# Load both CSV files
df1 = pd.read_csv('HSV Read Test.csv')
df2 = pd.read_csv('HSV Tif Read Test.csv')

# Check if shapes are the same
if df1.shape != df2.shape:
    print("Files have different shapes:", df1.shape, "vs", df2.shape)

# Compare data
comparison = df1.equals(df2)

if comparison:
    print("✅ The two CSV files are exactly the same.")
else:
    print("❌ The two CSV files are different.")
    
    # Optional: show differences
    diff = df1 != df2
    differences = df1[diff]
    print("Differences found at the following positions:")
    print(differences)
