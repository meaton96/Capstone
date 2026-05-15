import pandas as pd

def compare_csvs(old_file, new_file, tolerance=0.1):
    # Load data
    df_old = pd.read_csv(old_file)
    df_new = pd.read_csv(new_file)
    
    # Keys to match on
    keys = ['rule', 'seed', 'jobs', 'machines', 'total_ops', 'agvCount']
    
    # Merge
    merged = pd.merge(
        df_new, 
        df_old[keys + ['makespan']], 
        on=keys, 
        how='left', 
        suffixes=('_new', '_old')
    )
    
    # Calculate difference
    merged['diff'] = (merged['makespan_new'] - merged['makespan_old']).abs()
    
    # 1. Matches within tolerance
    valid = merged[merged['diff'] <= tolerance].copy()
    
    # 2. Matches that failed tolerance (The Diverging Rows)
    failed = merged[merged['diff'] > tolerance].copy()
    
    # 3. Rows in New that didn't exist in Old
    missing = merged[merged['makespan_old'].isna()].copy()
    
    # Print Summary
    print("-" * 30)
    print(f"COMPARISON SUMMARY")
    print("-" * 30)
    print(f"Total rows in New file: {len(df_new)}")
    print(f"--- Valid Matches: {len(valid)}")
    print(f"--- Mismatches:    {len(failed)}")
    print(f"--- Not in Old:    {len(missing)}")
    print("-" * 30)
    
    # Output diverging rows for easy comparison
    if not failed.empty:
        print("\nDETAILED MISMATCHES (Diverging Rows):")
        # Define columns to display: Keys + the values we care about
        display_cols = keys + ['makespan_old', 'makespan_new', 'diff']
        # Sort by diff descending so the biggest outliers are at the top
        print(failed[display_cols].sort_values(by='diff', ascending=False).to_string(index=False))
    else:
        print("\nNo mismatches found within the given tolerance.")
    
    return valid, failed

# Run the comparison
valid_df, failed_df = compare_csvs('results_brand_old.csv', 'baseline_results.csv', tolerance=0.1)