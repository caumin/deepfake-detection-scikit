import os
import subprocess
import argparse
import logging
import pandas as pd
import json
from tqdm import tqdm

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_command(command):
    """Runs a shell command and captures its output."""
    logging.info(f"Executing command: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True, encoding='utf-8')
        # Log stdout only if it's not excessively long
        if result.stdout and len(result.stdout) < 1000:
            logging.info(result.stdout)
        if result.stderr:
            logging.warning(result.stderr)
    except FileNotFoundError as e:
        logging.error(f"Error: Command not found. Ensure '{command[0]}' is in your PATH. Details: {e}")
        raise
    except subprocess.CalledProcessError as e:
        logging.error(f"Command '{' '.join(e.cmd)}' failed with exit code {e.returncode}.")
        logging.error(f"Stdout: {e.stdout}")
        logging.error(f"Stderr: {e.stderr}")
        raise

def main(args):
    """Main feature probing orchestration function."""
    dataset_name = os.path.basename(os.path.normpath(args.data_dir))
    
    # --- 1. Define Paths & Ensure Feature CSVs Exist ---
    # This script assumes features have been extracted by run_pipeline.py first.
    train_csv = os.path.join(args.output_dir, f"{dataset_name}_train_features.csv")
    test_csv = os.path.join(args.output_dir, f"{dataset_name}_test_features.csv")

    if not (os.path.exists(train_csv) and os.path.exists(test_csv)):
        logging.error(f"Feature files not found. Please run 'run_pipeline.py' for dataset '{dataset_name}' first.")
        logging.error(f"Expected train file: {train_csv}")
        logging.error(f"Expected test file: {test_csv}")
        return

    # --- 2. Define Feature Groups ---
    logging.info(f"Loading feature list from {train_csv}...")
    df = pd.read_csv(train_csv)
    all_columns = [col for col in df.columns if col not in ['path', 'label']]
    
    # Define feature groups based on prefixes/names from features.py
    feature_groups = {
        'Spectrum': [c for c in all_columns if c.startswith('spec_bin_')],
        'Distortion': [c for c in all_columns if c.startswith('spec_ratio_')],
        'Color': [c for c in all_columns if c.startswith('sat_') or c.startswith('corr_')],
        'Residuals': [c for c in all_columns if c.startswith('res_cov_')]
    }
    logging.info(f"Defined {len(feature_groups)} feature groups to probe.")

    # --- 3. Feature Probing Loop ---
    probe_base_dir = os.path.join(args.output_dir, f"feature_group_probes_{args.model}")
    models_dir = os.path.join(probe_base_dir, "models")
    reports_dir = os.path.join(probe_base_dir, "reports")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    results = []

    logging.info(f"--- Starting Feature Group Probing with model: {args.model.upper()} ---")
    for group_name, feature_list in tqdm(feature_groups.items(), desc="Probing feature groups"):
        
        if not feature_list:
            logging.warning(f"Feature group '{group_name}' is empty, skipping.")
            continue

        try:
            logging.info(f"--- Processing feature group: {group_name} ({len(feature_list)} features) ---")
            
            model_path = os.path.join(models_dir, f"{dataset_name}_{args.model}_{group_name}.joblib")
            report_dir = os.path.join(reports_dir, f"report_{dataset_name}_{args.model}_{group_name}")
            os.makedirs(report_dir, exist_ok=True)

            # --- Train a model on the feature group ---
            logging.info(f"Training {args.model} on feature group '{group_name}'...")
            train_cmd = [
                'python', 'train.py',
                '--csv', train_csv,
                '--model', args.model,
                '--out_model', model_path,
                '--test_size', '0.01', # Use minimal validation split
                '--features'
            ] + feature_list
            run_command(train_cmd)
            
            # --- Evaluate the model on the test set ---
            logging.info(f"Evaluating model for feature group '{group_name}'...")
            eval_cmd = [
                'python', 'eval.py',
                '--csv', test_csv,
                '--model', model_path,
                '--report_dir', report_dir
            ]
            run_command(eval_cmd)

            # --- Collect results from the report ---
            report_path = os.path.join(report_dir, 'report.json')
            if os.path.exists(report_path):
                with open(report_path, 'r') as f:
                    report_data = json.load(f)
                auroc = report_data.get('auroc', 0)
                accuracy = report_data.get('accuracy', 0)
                results.append({
                    'feature_group': group_name,
                    'auroc': auroc,
                    'accuracy': accuracy,
                    'feature_count': len(feature_list)
                })
                logging.info(f"Group '{group_name}' | AUROC: {auroc:.4f}, Accuracy: {accuracy:.4f}")
            else:
                logging.warning(f"Could not find report file for group '{group_name}' at {report_path}")

        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            logging.error(f"Pipeline failed for feature group '{group_name}'. Skipping. Error: {e}")
            continue
            
    # --- 4. Save Final Report ---
    if not results:
        logging.warning("No results were generated. Exiting without creating a report.")
        return

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='auroc', ascending=False)

    final_report_path = os.path.join(probe_base_dir, 'feature_group_performance_report.csv')
    logging.info(f"--- Saving final performance report to {final_report_path} ---")
    results_df.to_csv(final_report_path, index=False)
    
    # Print the full results table
    print("\n--- Feature Group Performance (by AUROC) ---")
    print(results_df.to_string(index=False))
    
    logging.info("--- Feature Group Probing Pipeline Finished ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Feature probing pipeline for image authenticity detection.")
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help="Path to the root dataset directory (e.g., 'data/CIFAKE')."
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help="Path to the output directory where features are stored (e.g., 'CIFAKE_output')."
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['linsvm', 'rbfsvm', 'rf', 'xgb', 'logreg'],
        help="The model to use for probing each feature."
    )
    args = parser.parse_args()
    main(args)
