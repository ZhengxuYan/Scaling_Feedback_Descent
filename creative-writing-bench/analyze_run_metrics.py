import json
import argparse
import sys
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from collections import defaultdict
import creative-writing-bench.text_metrics

def load_all_runs(file_path):
    """Load the entire runs JSON file."""
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File not found at {path}")
        sys.exit(1)
        
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file: {e}")
        sys.exit(1)

def get_run_data(all_data, run_id):
    """Get data for a specific run ID."""
    if run_id not in all_data:
        print(f"Error: Run ID '{run_id}' not found in the file.")
        print("Available run IDs (first 10):")
        for i, key in enumerate(all_data.keys()):
            if i >= 10:
                break
            print(f"  - {key}")
        sys.exit(1)
    return all_data[run_id]

def extract_responses(run_data):
    """Extract all model responses from the run data."""
    responses = []
    
    if "creative_tasks" not in run_data:
        print("Warning: 'creative_tasks' key not found in run data.")
        return responses

    # Structure: creative_tasks -> iteration_index -> prompt_id
    creative_tasks = run_data["creative_tasks"]
    
    for iter_idx, prompts in creative_tasks.items():
        if not isinstance(prompts, dict):
            continue
            
        for prompt_id, prompt_data in prompts.items():
            if not isinstance(prompt_data, dict):
                continue
                
            if "results_by_modifier" in prompt_data:
                results_by_modifier = prompt_data["results_by_modifier"]
                for modifier, result in results_by_modifier.items():
                    if "model_response" in result:
                        responses.append(result["model_response"])
                        
    return responses

def compute_metrics_for_run(run_id, responses):
    """Compute average metrics for a single run."""
    print(f"Analyzing {len(responses)} responses for run '{run_id}'...")
    
    if not responses:
        return {}

    aggregated_metrics = defaultdict(list)
    
    for i, text in enumerate(responses):
        try:
            metrics = text_metrics.compute_metrics(text)
            for key, value in metrics.items():
                aggregated_metrics[key].append(value)
        except Exception as e:
            print(f"Warning: Failed to compute metrics for response {i}: {e}")

    avg_metrics = {}
    for key, values in aggregated_metrics.items():
        if values:
            avg_metrics[key] = statistics.mean(values)
            
    return avg_metrics

def print_comparison_table(metrics_by_run):
    """Print a comparison table of metrics."""
    run_ids = list(metrics_by_run.keys())
    all_metric_keys = sorted(list(set(k for m in metrics_by_run.values() for k in m.keys())))
    
    # Calculate column widths
    metric_col_width = max(len(k) for k in all_metric_keys) + 2
    run_col_width = max(max(len(rid) for rid in run_ids), 10) + 2
    
    # Header
    header = f"{'Metric':<{metric_col_width}}"
    for run_id in run_ids:
        header += f"{run_id:<{run_col_width}}"
    
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    
    for metric in all_metric_keys:
        row = f"{metric:<{metric_col_width}}"
        base_val = None
        
        for i, run_id in enumerate(run_ids):
            val = metrics_by_run[run_id].get(metric, "N/A")
            if isinstance(val, float):
                formatted_val = f"{val:.4f}"
                if i == 0:
                    base_val = val
                    row += f"{formatted_val:<{run_col_width}}"
                else:
                    if base_val is not None and base_val != 0:
                        diff = ((val - base_val) / base_val) * 100
                        diff_str = f" ({diff:+.1f}%)"
                        row += f"{formatted_val}{diff_str:<{run_col_width - len(formatted_val)}}"
                    else:
                        row += f"{formatted_val:<{run_col_width}}"
            else:
                row += f"{val:<{run_col_width}}"
        print(row)
    print("=" * len(header))

def plot_percentage_difference(metrics_by_run, run_ids):
    """Generate a bar chart showing percentage difference relative to the first run."""
    baseline_run = run_ids[0]
    baseline_metrics = metrics_by_run[baseline_run]
    
    data = []
    # Skip the first run as it is the baseline (0% diff)
    for run_id in run_ids[1:]:
        metrics = metrics_by_run[run_id]
        for metric, value in metrics.items():
            base_val = baseline_metrics.get(metric, 0)
            if base_val != 0:
                diff_pct = ((value - base_val) / base_val) * 100
                data.append({"Run": run_id, "Metric": metric, "Difference (%)": diff_pct})
    
    if not data:
        print("No data for percentage difference plot.")
        return

    df = pd.DataFrame(data)
    
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    
    # Create barplot
    chart = sns.barplot(
        data=df, 
        x="Metric", 
        y="Difference (%)", 
        hue="Run",
        palette="tab10"
    )
    
    # Add a horizontal line at 0
    plt.axhline(0, color='black', linewidth=1)
    
    plt.title(f"Percentage Difference relative to {baseline_run}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Construct filename
    short_ids = [rid.split('__')[0] for rid in run_ids]
    filename = f"diff_plot_{'_vs_'.join(short_ids)}.png"
    filename = "".join(c for c in filename if c.isalnum() or c in ('_', '.', '-'))
    
    plt.savefig(filename, dpi=300)
    print(f"\nPercentage difference plot saved to: {filename}")

def plot_normalized_comparison(metrics_by_run, run_ids):
    """Generate a heatmap or normalized bar chart."""
    # Normalize each metric by the maximum value across all runs
    all_metrics = set()
    for m in metrics_by_run.values():
        all_metrics.update(m.keys())
        
    data = []
    for metric in all_metrics:
        # Find max value for this metric
        max_val = max(metrics_by_run[rid].get(metric, 0) for rid in run_ids)
        
        if max_val == 0:
            continue
            
        for run_id in run_ids:
            val = metrics_by_run[run_id].get(metric, 0)
            norm_val = val / max_val
            data.append({"Run": run_id, "Metric": metric, "Normalized Score": norm_val})
            
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    
    chart = sns.barplot(
        data=df,
        x="Metric",
        y="Normalized Score",
        hue="Run",
        palette="viridis"
    )
    
    plt.title("Normalized Metrics (Scaled to Max=1.0)")
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    short_ids = [rid.split('__')[0] for rid in run_ids]
    filename = f"norm_plot_{'_vs_'.join(short_ids)}.png"
    filename = "".join(c for c in filename if c.isalnum() or c in ('_', '.', '-'))
    
    plt.savefig(filename, dpi=300)
    print(f"Normalized plot saved to: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Analyze and compare text metrics for run IDs.")
    parser.add_argument("run_ids", nargs='+', help="One or more run IDs to analyze")
    parser.add_argument(
        "--file", 
        default="creative-writing-bench/creative_bench_runs.json",
        help="Path to the runs JSON file"
    )
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.file}...")
    all_data = load_all_runs(args.file)
    
    metrics_by_run = {}
    
    for run_id in args.run_ids:
        run_data = get_run_data(all_data, run_id)
        responses = extract_responses(run_data)
        metrics = compute_metrics_for_run(run_id, responses)
        metrics_by_run[run_id] = metrics
        
    print_comparison_table(metrics_by_run)
    
    if len(args.run_ids) > 1:
        # Generate both types of plots for better visualization
        plot_percentage_difference(metrics_by_run, args.run_ids)
        plot_normalized_comparison(metrics_by_run, args.run_ids)

if __name__ == "__main__":
    main()
