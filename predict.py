import os
import argparse
import joblib
import numpy as np
import cv2
from tqdm import tqdm
import glob
import pandas as pd

from features import extract_all_features#, get_feature_names # Removed get_feature_names

def main():
    parser = argparse.ArgumentParser(description="Predict if an image is real or fake using a trained model.")
    parser.add_argument('--model', type=str, required=True, help="Path to the trained model (.joblib file).")
    parser.add_argument('--input', type=str, required=True, help="Path to a single image or a directory of images.")
    # The following arguments should match the settings used for training
    parser.add_argument('--img_size', type=int, default=256, help="Size to resize images to.")
    parser.add_argument('--reencode_jpeg', type=int, default=95, help="JPEG quality for re-encoding. Set to 0 to disable.")
    parser.add_argument('--bins', type=int, default=128, help="Number of bins for 1D power spectrum (must match training).")

    args = parser.parse_args()

    # --- Load Model ---
    print(f"Loading model from {args.model}...")
    try:
        saved_object = joblib.load(args.model)
        model = saved_object['model']
        training_feature_names = saved_object['feature_names'] # Get feature names from the saved object
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- Prepare Input ---
    if os.path.isdir(args.input):
        image_paths = glob.glob(os.path.join(args.input, '**', '*.jpg'), recursive=True) + \
                      glob.glob(os.path.join(args.input, '**', '*.png'), recursive=True)
        print(f"Found {len(image_paths)} images to predict.")
    elif os.path.isfile(args.input):
        image_paths = [args.input]
    else:
        print(f"Error: Input path '{args.input}' not found.")
        return

    if not image_paths:
        print("No images found to process.")
        return

    # --- Predict ---
    reencode_jpeg_quality = args.reencode_jpeg if args.reencode_jpeg > 0 else None
    results = []

    for path in tqdm(image_paths, desc="Predicting"):
        try:
            img_array = np.fromfile(path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                print(f"Warning: Could not read {path}")
                continue

            # Preprocessing (must be identical to training)
            img = cv2.resize(img, (args.img_size, args.img_size), interpolation=cv2.INTER_AREA)
            if reencode_jpeg_quality:
                _, img_encoded = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), reencode_jpeg_quality])
                img = cv2.imdecode(img_encoded, cv2.IMREAD_COLOR)
            
            # Feature Extraction (must be identical to training)
            # Make sure to pass correct bins to extract_all_features
            feature_vector_dict = extract_all_features(img, spec_bins=args.bins, color_bins=32, residual_mode='highpass') # default color_bins=32, residual_mode='highpass'
            
            # Convert feature_vector_dict to a pandas Series, then to DataFrame, ensuring order of features
            feature_series = pd.Series(feature_vector_dict)
            
            # Align features with the ones the model was trained on
            # This is crucial for models like XGBoost that care about feature names/order
            # We need to make sure the feature_series contains all training_feature_names
            # If some features are missing in the extracted feature_series, they will be NaN in the DataFrame,
            # which might cause issues. However, if extract_all_features is consistent, this should be fine.
            df = pd.DataFrame([feature_series[training_feature_names].values], columns=training_feature_names)

            # Prediction
            prediction = model.predict(df)[0]
            probability = model.predict_proba(df)[0]
            
            label = "REAL" if prediction == 1 else "FAKE"
            
            results.append({
                'path': path,
                'prediction': label,
                'prob_fake': probability[0],
                'prob_real': probability[1]
            })

        except Exception as e:
            print(f"Error processing {path}: {e}")

    # --- Display Results ---
    print("\n--- Prediction Results ---")
    for res in results:
        print(f"File: {os.path.basename(res['path']):<30} | Prediction: {res['prediction']:<5} | Confidence (REAL): {res['prob_real']:.2%}")

if __name__ == '__main__':
    main()
