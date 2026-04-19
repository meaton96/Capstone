import pandas as pd
import glob
import os

def combine_csv_files(output_filename="agv_sweep.csv"):
    # 1. Get a list of all CSV files in the current directory
    # If your files are in a specific folder, use 'path/to/folder/*.csv'
    all_files = glob.glob("*.csv")
    
    # Optional: Filter out the output file if it already exists to avoid recursion
    all_files = [f for f in all_files if f != output_filename]

    if not all_files:
        print("No CSV files found in the directory.")
        return

    print(f"Found {len(all_files)} files. Combining...")

    # 2. Use a list comprehension to read all CSVs into DataFrames
    df_list = [pd.read_csv(filename) for filename in all_files]

    # 3. Concatenate all DataFrames in the list
    # ignore_index=True ensures the new index is continuous (0, 1, 2...)
    combined_df = pd.concat(df_list, ignore_index=True)

    # 4. Export the result to a new CSV file
    combined_df.to_csv(output_filename, index=False)
    
    print(f"Success! Files combined into: {output_filename}")

if __name__ == "__main__":
    combine_csv_files()