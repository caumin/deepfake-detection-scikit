import os
import glob
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def visualize_reports(reports_dir="Gemini_output"):
    """
    Finds all report.json files, extracts key metrics, and creates a
    comparative visualization using seaborn.
    """
    report_paths = glob.glob(os.path.join(reports_dir, "*", "report.json"))

    if not report_paths:
        print(f"No report.json files found in {reports_dir}. Exiting.")
        return

    print(f"Found {len(report_paths)} reports. Processing...")

    all_metrics = []
    for path in report_paths:
        try:
            # Extract model name from path, e.g., 'report_gemini_linsvm' -> 'linsvm'
            model_name = os.path.basename(os.path.dirname(path)).replace("report_gemini_", "")

            with open(path, 'r', encoding='utf-8') as f:
                report = json.load(f)

            # Handle case where AUROC might not be available
            auroc = report.get("auroc")
            if isinstance(auroc, str) and "N/A" in auroc:
                auroc = np.nan
            else:
                auroc = float(auroc) if auroc is not None else np.nan

            metrics = {
                "model": model_name.upper(),
                "Accuracy": report.get("accuracy"),
                "AUROC": auroc,
                "F1-score (Weighted)": report.get("classification_report", {}).get("weighted avg", {}).get("f1-score")
            }
            all_metrics.append(metrics)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Could not process {path}: {e}")
            continue
    
    if not all_metrics:
        print("No valid metrics could be extracted. Exiting.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(all_metrics)

    # Melt the DataFrame for easy plotting with seaborn
    df_melted = df.melt(id_vars="model", var_name="Metric", value_name="Score")
    df_melted.dropna(inplace=True) # Drop metrics that were not available (e.g., all AUROCs were N/A)
    
    if df_melted.empty:
        print("No valid data to plot after cleaning. Exiting.")
        return

    # Create the plot
    plt.figure(figsize=(12, 8))
    barplot = sns.barplot(data=df_melted, x="model", y="Score", hue="Metric", palette="viridis")
    
    # Add score labels on top of bars
    for p in barplot.patches:
        barplot.annotate(format(p.get_height(), '.3f'),
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha = 'center', va = 'center',
                       xytext = (0, 9),
                       textcoords = 'offset points')

    plt.title("Model Performance Comparison (Gemini Dataset)", fontsize=16)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(reports_dir, "model_comparison.png")
    plt.savefig(output_path, dpi=300)
    print(f"Visualization saved to: {output_path}")

if __name__ == "__main__":
    visualize_reports()
