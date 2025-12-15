import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os

def analyze_correlation(csv_path, output_dir, threshold=0.9):
    """
    Analyzes and visualizes the correlation matrix of features from a CSV file.

    Args:
        csv_path (str): Path to the feature CSV file.
        output_dir (str): Directory to save the output heatmap image.
        threshold (float): Correlation threshold to report on.
    """
    print(f"Loading features from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return

    # Keep only numeric columns for correlation analysis (features)
    features_df = df.select_dtypes(include=np.number)
    
    # The 'label' column is not a feature
    if 'label' in features_df.columns:
        features_df = features_df.drop(columns=['label'])

    if features_df.shape[1] < 2:
        print("Error: Not enough numeric feature columns to perform correlation analysis.")
        return

    print(f"Calculating correlation matrix for {features_df.shape[1]} features...")
    corr_matrix = features_df.corr()

    # --- Find and print highly correlated pairs ---
    print(f"\n--- Highly Correlated Feature Pairs (threshold > {threshold}) ---")
    # Create a boolean matrix for correlations above the threshold
    high_corr_mask = (corr_matrix.abs() > threshold) & (corr_matrix.abs() < 1.0)
    high_corr_pairs = corr_matrix[high_corr_mask].stack().reset_index()
    high_corr_pairs.columns = ['Feature 1', 'Feature 2', 'Correlation']

    # Remove duplicate pairs (e.g., (A, B) and (B, A))
    high_corr_pairs['sorted_features'] = high_corr_pairs.apply(lambda row: tuple(sorted((row['Feature 1'], row['Feature 2']))), axis=1)
    high_corr_pairs = high_corr_pairs.drop_duplicates(subset='sorted_features').drop(columns='sorted_features')

    if high_corr_pairs.empty:
        print("No feature pairs found above the specified threshold.")
    else:
        for index, row in high_corr_pairs.iterrows():
            print(f"- {row['Feature 1']} & {row['Feature 2']}: {row['Correlation']:.4f}")
    print("----------------------------------------------------")


    # --- Generate and save heatmap ---
    print("Generating correlation heatmap...")
    plt.figure(figsize=(20, 18))
    sns.heatmap(corr_matrix, cmap='viridis', annot=False) # Annot is false for readability with many features
    plt.title('Feature Correlation Matrix', fontsize=16)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    heatmap_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    print(f"Heatmap saved to {heatmap_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze feature correlation from a CSV file.')
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to the input CSV file containing features.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='out',
        help='Directory to save the output correlation heatmap. Defaults to "out".'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.9,
        help='Correlation coefficient threshold to report highly correlated pairs. Defaults to 0.9.'
    )
    args = parser.parse_args()

    try:
        import seaborn as sns
    except ImportError:
        print("Seaborn is not installed. Please install it using: pip install seaborn")
    else:
        analyze_correlation(args.csv, args.output_dir, args.threshold)
