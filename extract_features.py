import os
import glob
import numpy as np
import cv2
from tqdm import tqdm
import argparse
import pandas as pd
from joblib import Parallel, delayed
from features import extract_all_features

def process_image(path, label_int, img_size, reencode_jpeg, spec_bins, color_bins, residual_mode):
    """
    Worker function for a single image: load, preprocess, and extract features.
    Returns a dictionary with features, label, and path.
    """
    try:
        img_array = np.fromfile(path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return None

        # 1. Preprocessing (convert to RGB from BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)

        if reencode_jpeg:
            _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), reencode_jpeg])
            img_final = cv2.imdecode(img_encoded, cv2.IMREAD_COLOR)
            img_final = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)
        else:
            img_final = img_resized

        # 2. Feature Extraction
        feature_dict = extract_all_features(img_final, spec_bins=spec_bins, color_bins=color_bins, residual_mode=residual_mode)
        
        # 3. Combine with metadata
        feature_dict['label'] = label_int
        feature_dict['path'] = path
        return feature_dict

    except Exception as e:
        print(f"Skipping {path} due to error: {e}")
        return None

def extract_features_from_dir(real_dir, fake_dir, img_size=256, reencode_jpeg=None, spec_bins=128, color_bins=32, n_jobs=-1, residual_mode='denoise'):
    """
    Extracts features from all images in parallel using joblib.
    """
    tasks = []
    
    image_dirs = {'REAL': (real_dir, 1), 'FAKE': (fake_dir, 0)}

    for label_str, (directory, label_int) in image_dirs.items():
        print(f"Preparing tasks for {label_str} images from: {directory}")
        image_paths = glob.glob(os.path.join(directory, '**', '*.jpg'), recursive=True) + \
                      glob.glob(os.path.join(directory, '**', '*.png'), recursive=True)

        if not image_paths:
            print(f"Warning: No images found in {directory}")
            continue

        for path in image_paths:
            tasks.append(delayed(process_image)(path, label_int, img_size, reencode_jpeg, spec_bins, color_bins, residual_mode))

    print(f"\nExtracting features from {len(tasks)} images using {n_jobs if n_jobs > 0 else 'all'} CPU cores...")
    results = Parallel(n_jobs=n_jobs)(tqdm(tasks))
    
    # Filter out None results from failed images
    valid_results = [res for res in results if res is not None]
    
    if not valid_results:
        print("Error: No features were extracted. Check image paths and file integrity.")
        return pd.DataFrame()
            
    # Convert list of dictionaries directly to a DataFrame
    df = pd.DataFrame.from_records(valid_results)
    
    # Reorder columns to have 'path' and 'label' first
    label_col = df.pop('label')
    path_col = df.pop('path')
    df.insert(0, 'path', path_col)
    df.insert(1, 'label', label_col)
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Extract features from real and fake image datasets.")
    parser.add_argument('--real_dir', type=str, required=True, help="Directory for real images.")
    parser.add_argument('--fake_dir', type=str, required=True, help="Directory for fake images.")
    parser.add_argument('--out_csv', type=str, default='out/features.csv', help="Path to save the output CSV file.")
    parser.add_argument('--img_size', type=int, default=256, help="Size to resize images to (e.g., 256).")
    parser.add_argument('--reencode_jpeg', type=int, default=95, help="JPEG quality for re-encoding (e.g., 95). Set to 0 to disable.")
    parser.add_argument('--bins', type=int, default=64, help="Number of bins for the 1D power spectrum feature.")
    parser.add_argument('--color_bins', type=int, default=16, help="Number of bins for the color saturation histogram.")
    parser.add_argument('--n_jobs', type=int, default=-1, help="Number of parallel jobs to run (-1 uses all available cores).")
    parser.add_argument('--residual_mode', type=str, default='highpass', choices=['denoise', 'highpass'], help="Method for noise residual calculation ('denoise' is slow, 'highpass' is fast).")
    
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    reencode_jpeg_quality = args.reencode_jpeg if args.reencode_jpeg > 0 else None

    features_df = extract_features_from_dir(
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        img_size=args.img_size,
        reencode_jpeg=reencode_jpeg_quality,
        spec_bins=args.bins,
        color_bins=args.color_bins,
        n_jobs=args.n_jobs,
        residual_mode=args.residual_mode
    )
    
    print(f"\nSaving {len(features_df)} feature vectors to {args.out_csv}...")
    if not features_df.empty:
        features_df.to_csv(args.out_csv, index=False)
    print("Done.")

if __name__ == '__main__':
    main()
