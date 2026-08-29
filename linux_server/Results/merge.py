import argparse
import glob
import os
import pandas as pd


def merge_csv_group(search_pattern, output_path):
    """Finds all CSVs matching the pattern, merges them, and saves the result."""
    # Find all files matching the pattern in the directory
    all_files = sorted(glob.glob(search_pattern))

    if not all_files:
        print(f"⚠️  No files found matching: {search_pattern}")
        return

    print(f"📦 Found {len(all_files)} files for pattern: {search_pattern}")

    # Read and concatenate all CSVs in the group
    df_list = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            filename = os.path.basename(file).lower()
            
            # Catch ANY file across ALL categories that has 'random' in the name
            if "random" in filename:
                if "rule" in df.columns:
                    df["rule"] = "random"
                    print(f"🔄 FORCED 'rule' column to 'random' for: {os.path.basename(file)}")
                else:
                    # If the column isn't there, create it so it aligns during concat
                    df["rule"] = "random"
                    print(f"➕ Created 'rule' column and set to 'random' for: {os.path.basename(file)}")
                    
            df_list.append(df)
        except Exception as e:
            print(f"❌ Error reading {file}: {e}")

    if df_list:
        merged_df = pd.concat(df_list, ignore_index=True)
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Save the merged file
        merged_df.to_csv(output_path, index=False)
        print(f"✅ Successfully created: {output_path}")
    print("-" * 50)


def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(
        description="Merge sets of simulation benchmark CSV files."
    )
    parser.add_argument(
        "-dir",
        required=False,
        help="Path to the directory containing the source CSV files.",
        default=os.path.join(os.getcwd(), "generated")
    )
    parser.add_argument(
        "-out",
        required=False,
        help="Path to the directory where merged CSVs should be saved.",
        default=os.path.join(os.getcwd(), "merged")
    )

    args = parser.parse_args()

    # Define the 5 target patterns and their respective output filenames
    categories = {
        "agv_performance_*.csv": "agv_performance.csv",
        "results_*.csv": "results.csv",
        "segment_congestion_*.csv": "segment_congestion.csv",
        "machine_utilization_*.csv": "machine_utilization.csv",
        "job_operations_*.csv": "job_operations.csv",
        "throughput_*.csv": "throughput.csv",
        "job_completions_*.csv": "job_completions.csv"
    }

    print(f"\n🚀 Starting merge process...")
    print(f"📂 Source Directory: {args.dir}")
    print(f"💾 Output Directory: {args.out}\n" + "=" * 50)

    # Process each category
    for pattern, output_name in categories.items():
        full_pattern = os.path.join(args.dir, pattern)
        full_output = os.path.join(args.out, output_name)
        merge_csv_group(full_pattern, full_output)

    print("🎉 All merging tasks completed!")


if __name__ == "__main__":
    main()