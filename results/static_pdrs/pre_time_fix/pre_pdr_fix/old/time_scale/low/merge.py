import glob
import pandas as pd

# Define the glob pattern to match your files
file_pattern = "results_bm_*.csv"

# Find all files matching the pattern
csv_files = glob.glob(file_pattern)

if not csv_files:
    print(f"No files found matching pattern: {file_pattern}")
else:
    print(f"Found {len(csv_files)} files. Combining...")

    # Read and concatenate all CSVs into a single DataFrame
    # (Pandas will automatically align the columns based on your schema)
    combined_df = pd.concat(
        [pd.read_csv(file) for file in csv_files], ignore_index=True
    )

    # Preview the combined data
    print("\nSuccess! Combined DataFrame shape:", combined_df.shape)
    print(combined_df.head())

    # Optional: Save the combined dataframe to a new CSV
    combined_df.to_csv("combined_results_bm.csv", index=False)