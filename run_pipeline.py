import os
import subprocess
import argparse
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_command(command):
    """Runs a shell command and lets its output stream to the console."""
    logging.info(f"Executing command: {' '.join(command)}")
    try:
        # The subprocess will inherit stdout/stderr, so its output will be displayed directly.
        subprocess.run(command, check=True, text=True, encoding='utf-8')
    except FileNotFoundError as e:
        logging.error(f"Error: Command not found. Ensure '{command[0]}' is in your PATH. Details: {e}")
        raise
    except subprocess.CalledProcessError as e:
        # With capture_output=False, stdout/stderr are not captured in the exception object.
        # The output would have been streamed to the console directly.
        logging.error(f"Command '{' '.join(command)}' failed with exit code {e.returncode}.")
        raise

def find_real_fake_dirs(base_path):
    """Finds 'real' and 'fake' directories, accommodating case variations."""
    # Check for lowercase 'real'/'fake'
    real_dir = os.path.join(base_path, 'real')
    fake_dir = os.path.join(base_path, 'fake')
    if os.path.isdir(real_dir) and os.path.isdir(fake_dir):
        return real_dir, fake_dir

    # Check for uppercase 'REAL'/'FAKE'
    real_dir_upper = os.path.join(base_path, 'REAL')
    fake_dir_upper = os.path.join(base_path, 'FAKE')
    if os.path.isdir(real_dir_upper) and os.path.isdir(fake_dir_upper):
        return real_dir_upper, fake_dir_upper

    raise FileNotFoundError(f"Could not find 'real'/'fake' or 'REAL'/'FAKE' directories in {base_path}")


def main(args):
    """Main pipeline orchestration function."""
    dataset_name = os.path.basename(os.path.normpath(args.data_dir))
    output_dir = f"{dataset_name}_output"
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Created output directory: {output_dir}")

    # Define paths for train/test features
    train_csv = os.path.join(output_dir, f"{dataset_name}_train_features.csv")
    test_csv = os.path.join(output_dir, f"{dataset_name}_test_features.csv")

    # --- 1. Feature Extraction ---
    # Check if data is pre-split or needs splitting
    train_base_path = os.path.join(args.data_dir, 'train')
    test_base_path = os.path.join(args.data_dir, 'test')
    is_pre_split = os.path.isdir(train_base_path) and os.path.isdir(test_base_path)

    skip_extraction = False
    if os.path.exists(train_csv) and os.path.exists(test_csv) and not args.force_extract:
        logging.info("Feature files already exist and --force-extract not specified. Skipping feature extraction.")
        skip_extraction = True

    if not skip_extraction:
        try:
            if is_pre_split:
                logging.info("--- Pre-split train/test folders found. Extracting features separately. ---")
                # For training data
                real_train_dir, fake_train_dir = find_real_fake_dirs(train_base_path)
                extract_cmd_train = [
                    'python', 'extract_features.py',
                    '--real_dir', real_train_dir, '--fake_dir', fake_train_dir,
                    '--out_csv', train_csv, '--bins', str(args.bins),
                    '--color_bins', str(args.color_bins), '--residual_mode', args.residual_mode,
                    '--img_size', str(args.img_size)
                ]
                run_command(extract_cmd_train)

                # For testing data
                real_test_dir, fake_test_dir = find_real_fake_dirs(test_base_path)
                extract_cmd_test = [
                    'python', 'extract_features.py',
                    '--real_dir', real_test_dir, '--fake_dir', fake_test_dir,
                    '--out_csv', test_csv, '--bins', str(args.bins),
                    '--color_bins', str(args.color_bins), '--residual_mode', args.residual_mode,
                    '--img_size', str(args.img_size)
                ]
                run_command(extract_cmd_test)
            else:
                logging.info("--- No pre-split folders found. Extracting all features and then splitting. ---")
                real_dir, fake_dir = find_real_fake_dirs(args.data_dir)
                all_features_csv = os.path.join(output_dir, f"{dataset_name}_all_features.csv")
                
                # Extract features for the entire dataset
                extract_cmd_all = [
                    'python', 'extract_features.py',
                    '--real_dir', real_dir, '--fake_dir', fake_dir,
                    '--out_csv', all_features_csv, '--bins', str(args.bins),
                    '--color_bins', str(args.color_bins), '--residual_mode', args.residual_mode,
                    '--img_size', str(args.img_size)
                ]
                run_command(extract_cmd_all)

                # Load the full feature set and split it
                logging.info(f"Loading features from {all_features_csv} for splitting...")
                df = pd.read_csv(all_features_csv)
                
                train_df, test_df = train_test_split(
                    df, 
                    test_size=0.2, # Standard 80/20 split
                    random_state=42,
                    stratify=df['label'] # Preserve class distribution
                )
                
                logging.info(f"Saving split features to {train_csv} and {test_csv}...")
                train_df.to_csv(train_csv, index=False)
                test_df.to_csv(test_csv, index=False)
                os.remove(all_features_csv) # Clean up the combined file
                logging.info("Splitting complete.")

            logging.info("--- Feature Extraction Complete ---")

        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            logging.error(f"Failed during feature extraction. Aborting pipeline. Error: {e}")
            return
    
    # --- 2. & 3. Training and Evaluation Loop ---
    # The rest of the script remains the same as it consumes train_csv and test_csv
    supported_models = ['linsvm', 'rbfsvm', 'xgb', 'rf']
    logging.info(f"--- Starting Training and Evaluation for models: {supported_models} ---")

    for model_name in supported_models:
        try:
            logging.info(f"--- Processing model: {model_name.upper()} ---")
            
            model_path = os.path.join(output_dir, f"{dataset_name}_{model_name}.joblib")
            report_dir = os.path.join(output_dir, f"report_{dataset_name}_{model_name}")
            os.makedirs(report_dir, exist_ok=True)

            # --- Training ---
            skip_training = False
            if os.path.exists(model_path) and not args.force_train:
                logging.info(f"Model file already exists: {model_path}. Skipping training.")
                skip_training = True

            if not skip_training:
                logging.info(f"Training {model_name}...")
                train_cmd = [
                    'python', 'train.py',
                    '--csv', train_csv,
                    '--model', model_name,
                    '--out_model', model_path
                ]
                run_command(train_cmd)
            
            # --- Evaluation ---
            logging.info(f"Evaluating {model_name}...")
            eval_cmd = [
                'python', 'eval.py',
                '--csv', test_csv,
                '--model', model_path,
                '--report_dir', report_dir
            ]
            
            # Add feature importance flag for supported models
            if model_name in ['rf', 'xgb']:
                eval_cmd.append('--feature-importance')

            # Add decision boundary plot arguments
            eval_cmd.extend(['--plot-decision-boundary', '--train-csv', train_csv])

            run_command(eval_cmd)
            logging.info(f"--- Finished processing model: {model_name.upper()} ---")

        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            logging.error(f"Pipeline failed for model '{model_name}'. Skipping to next model. Error: {e}")
            continue
            
    logging.info("--- Pipeline Finished ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Orchestration pipeline for image authenticity detection.")
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help="Path to the dataset directory (e.g., 'data/CIFAKE')."
    )
    parser.add_argument(
        '--feature-importance',
        action='store_true',
        help="If set, generate feature importance plots for tree-based models (rf, xgb)."
    )
    parser.add_argument(
        '--force-train',
        action='store_true',
        help="If set, retrain all models even if model files already exist."
    )
    parser.add_argument(
        '--force-extract',
        action='store_true',
        help="If set, re-extract features even if feature files already exist."
    )
    parser.add_argument(
        '--bins',
        type=int,
        default=32,
        help="Number of bins for the 1D power spectrum feature."
    )
    parser.add_argument(
        '--color_bins',
        type=int,
        default=32,
        help="Number of bins for the color saturation histogram."
    )
    parser.add_argument(
        '--residual_mode',
        type=str,
        default='denoise',
        choices=['denoise', 'highpass'],
        help="Method for noise residual calculation ('denoise' is slow, 'highpass' is fast)."
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=224,
        help="Size to resize images to (e.g., 224)."
    )
    args = parser.parse_args()
    main(args)
