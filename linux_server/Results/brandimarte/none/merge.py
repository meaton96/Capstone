import glob
import os
import pandas as pd

# Define the 4 glob patterns and their respective identifier keys
patterns = {
    "agv_performance": "agv_performance_bm_*.csv",
    "machine_utilization": "machine_utilization_bm_*.csv",
    "results": "results_bm_*.csv",
    "segment_congestion": "segment_congestion_bm_*.csv",
}

# Loop through each pattern and merge the files
for key, pattern in patterns.items():
    print(f"Processing group: {key} ({pattern})")

    # Find files matching the current pattern
    csv_files = glob.glob(pattern)

    if not csv_files:
        print(f"  --> Warning: No files found for pattern '{pattern}'\n")
        continue

    print(f"  --> Found {len(csv_files)} files. Concatenating...")

    # Combine all matching CSVs into one DataFrame
    combined_df = pd.concat(
        [pd.read_csv(file) for file in csv_files], ignore_index=True
    )

    # Generate the output filename
    output_filename = f"combined_{key}.csv"

    # Save the merged dataframe to a new CSV file
    combined_df.to_csv(output_filename, index=False)
    print(
        f"  --> Success! Saved {combined_df.shape[0]} rows into '{output_filename}'\n"
    )

print("All merging operations complete!")