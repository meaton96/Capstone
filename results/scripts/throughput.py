import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_summarize(csv_path, smooth_window=30):
    """Loads CSV, applies aggressive smoothing, and generates summary stats."""
    df = pd.read_csv(csv_path)
    
    # --- HEAVY SMOOTHING ---
    # Increased window to 30 intervals to show macro-trends rather than second-to-second noise
    df['throughput_smooth'] = df.groupby(['instance', 'rule', 'seed'])['throughput_per_min'].transform(
        lambda x: x.rolling(window=smooth_window, min_periods=1).mean()
    )
    
    run_summary = df.groupby(['instance', 'rule', 'seed']).agg(
        makespan=('makespan', 'max'),
        total_jobs=('cumulative_completed', 'max'),
        avg_wip=('work_in_progress', 'mean')
    ).reset_index()
    
    return df, run_summary

def generate_figures(df, run_summary, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # ==========================================
    # 1. THE AGGREGATED BASELINE (All Rules Combined)
    # ==========================================
    print("Generating Aggregated Baseline Plot...")
    # By dropping the 'hue' argument, Seaborn automatically averages everything 
    # (all rules and seeds) to show the general behavior of the instance.
    g = sns.relplot(
        data=df, x='window_end', y='throughput_smooth',
        col='instance', col_wrap=2, kind='line', 
        facet_kws={'sharex': False, 'sharey': False},
        height=4, aspect=1.8, errorbar=None, color='black', linewidth=2
    )
    g.fig.suptitle('Baseline Throughput Curve (Averaged Across All Rules & Seeds)', y=1.02)
    g.set_axis_labels('Simulation Time (seconds)', 'Throughput (Jobs / Min)')
    g.savefig(os.path.join(output_dir, '00_aggregated_throughput_baseline.png'), dpi=300, bbox_inches='tight')
    plt.close()


    # ==========================================
    # 2. INDIVIDUAL WIDE PLOTS PER INSTANCE
    # ==========================================
    print("Generating Individual Instance Breakouts (Wide X-Axis)...")
    instances = df['instance'].unique()
    
    for inst in instances:
        # Filter data for just this instance
        inst_df = df[df['instance'] == inst]
        
        # --- Throughput Plot ---
        plt.figure(figsize=(14, 6)) # Wide format to stretch the X-axis
        sns.lineplot(
            data=inst_df, x='window_end', y='throughput_smooth', 
            hue='rule', errorbar=None, linewidth=1.5
        )
        plt.title(f'Smoothed Throughput Comparison - {inst}', fontsize=16)
        plt.xlabel('Simulation Time (seconds)', fontsize=12)
        plt.ylabel(f'Throughput (Jobs / Min) - {30}-min rolling avg', fontsize=12)
        
        # Move the legend outside the plot so it doesn't cover data
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title='Dispatching Rule')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'throughput_isolated_{inst}.png'), dpi=300)
        plt.close()
        
        # --- WIP Plot ---
        plt.figure(figsize=(14, 6))
        sns.lineplot(
            data=inst_df, x='window_end', y='work_in_progress', 
            hue='rule', errorbar=None, linewidth=1.5
        )
        plt.title(f'Work In Progress (WIP) Drawdown - {inst}', fontsize=16)
        plt.xlabel('Simulation Time (seconds)', fontsize=12)
        plt.ylabel('WIP (Active Jobs)', fontsize=12)
        
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title='Dispatching Rule')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'wip_isolated_{inst}.png'), dpi=300)
        plt.close()


    # ==========================================
    # 3. MAKESPAN DISTRIBUTION (Kept as is)
    # ==========================================
    print("Generating Faceted Makespan Boxplots...")
    g = sns.catplot(
        data=run_summary, x='rule', y='makespan',
        col='instance', col_wrap=2, kind='box', sharey=False,
        height=4, aspect=1.5
    )
    g.fig.suptitle('Makespan Distribution by Rule', y=1.02)
    g.set_axis_labels('Dispatching Rule', 'Makespan')
    g.set_xticklabels(rotation=45)
    g.savefig(os.path.join(output_dir, '00_makespan_faceted.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate isolated, wide figures for DJFSP throughput.")
    parser.add_argument("csv_file", type=str, help="Path to the input CSV file")
    parser.add_argument("-o", "--outdir", type=str, default=".", help="Directory to save the figures")
    
    args = parser.parse_args()
    
    try:
        raw_df, run_stats = load_and_summarize(args.csv_file)
        generate_figures(raw_df, run_stats, args.outdir)
        print(f"\nSuccess! Visualizations saved to: {os.path.abspath(args.outdir)}")
    except FileNotFoundError:
        print(f"Error: Could not find '{args.csv_file}'. Please check the file path.")