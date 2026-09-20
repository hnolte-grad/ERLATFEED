import pandas as pd
import os

# File paths
input_file = r"C:\_temp workspace\1ER-LAT-FEED\data\feed\analyzed-feedevents\analyzed-feedevents_MASTER.csv"
output_file = r"C:\_temp workspace\1ER-LAT-FEED\data\3d model\surface area\randomselections.csv"

# Load data
df = pd.read_csv(input_file)

# Make sure the required columns exist
required_cols = ['sI', 'eI', 'depID', 'feedID']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in input CSV.")

# Randomly sample 10% of rows
sampled_df = df.sample(frac=0.1, random_state=42)  # random_state for reproducibility

# Keep only needed columns
sampled_df = sampled_df[required_cols]

# Save to CSV
os.makedirs(os.path.dirname(output_file), exist_ok=True)
sampled_df.to_csv(output_file, index=False)

print(f"Random 10% selection saved to: {output_file}")
