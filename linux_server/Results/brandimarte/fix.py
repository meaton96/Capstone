import pandas as pd
import glob

def salvage_data(file_pattern, dedupe_columns, output_name):
    files = glob.glob(file_pattern)
    if not files:
        print(f"No files found matching {file_pattern}")
        return
        
    all_data = []
    for f in files:
        df = pd.read_csv(f)
        
        # 1. Fix the double-logging bug using the specific columns for this dataset
        df = df.drop_duplicates(subset=dedupe_columns)
        
        # 2. Fix the "Random" identity crisis
        if "random" in f:
            df['rule'] = "random"
            
        all_data.append(df)

    # Combine and save
    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv(output_name, index=False)
    print(f"Success: Recovered {len(final_df)} clean rows and saved to {output_name}")

# --- 1. Salvage Episode Results ---
print("Processing episode data...")
salvage_data(
    file_pattern="baseline_results_bm_*.csv",
    dedupe_columns=['timestamp', 'instance', 'seed', 'makespan'],
    output_name="salvaged_clean_results.csv"
)

# --- 2. Salvage Machine Utilization ---
# Notice we add 'machine_id' to the dedupe list so we don't accidentally delete unique machines 
# that share the same timestamp for a given episode.
print("\nProcessing machine utilization data...")
salvage_data(
    file_pattern="machine_utilization_bm_*.csv",
    dedupe_columns=['timestamp', 'instance', 'seed', 'machine_id'], 
    output_name="salvaged_clean_machine_utilization.csv"
)