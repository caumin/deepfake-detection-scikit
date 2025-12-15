import pandas as pd
import argparse
import joblib
import os
from features import get_feature_names

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

from sklearn.kernel_approximation import Nystroem
from sklearn.svm import SVC, LinearSVC

# ... (rest of the imports) ...

def get_model(model_name, n_jobs=-1):
    """Returns a scikit-learn model instance based on its name."""
    if model_name == 'linsvm':
        # Use LinearSVC for much faster training on large datasets compared to SVC(kernel='linear')
        return make_pipeline(StandardScaler(), LinearSVC(C=1, class_weight='balanced', random_state=42, dual=True))
    elif model_name == 'rbfsvm':
        # Use Nystroem kernel approximation for a much faster, scalable RBF SVM
        return make_pipeline(StandardScaler(), Nystroem(kernel='rbf', random_state=42, n_components=200), LinearSVC(random_state=42, dual=True))
    elif model_name == 'rf':
        return make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_leaf=2, random_state=42, n_jobs=n_jobs))
    elif model_name == 'xgb':
        return XGBClassifier(n_estimators=800, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=42, n_jobs=n_jobs)
    elif model_name == 'logreg':
        return make_pipeline(StandardScaler(), LogisticRegression(random_state=42, n_jobs=n_jobs))
    else:
        raise ValueError(f"Model '{model_name}' not supported.")

def main():
    parser = argparse.ArgumentParser(description="Train a model on extracted features.")
    parser.add_argument('--csv', type=str, required=True, help="Path to the features CSV file.")
    parser.add_argument('--model', type=str, default='rf', choices=['linsvm', 'rbfsvm', 'rf', 'xgb', 'logreg'], help="Model to train.")
    parser.add_argument('--out_model', type=str, default='out/model.joblib', help="Path to save the trained model.")
    parser.add_argument('--test_size', type=float, default=0.2, help="Proportion of the dataset to include in the test split.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--features', type=str, nargs='+', default=None, help="Train on one or more specified feature columns.")
    
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)

    print(f"Loading features from {args.csv}...")
    df = pd.read_csv(args.csv)

    if args.features:
        print(f"Training on specified feature group with {len(args.features)} feature(s).")
        # Check if all specified features exist in the dataframe
        missing_features = set(args.features) - set(df.columns)
        if missing_features:
            raise ValueError(f"The following features were not found in the CSV: {', '.join(missing_features)}")
        feature_columns = args.features
    else:
        # Dynamically get feature columns from the features module
        print("Training on all features.")
        feature_columns = get_feature_names(feature_size_spec1d=len([c for c in df.columns if c.startswith('spec_bin_')]))
    
    X = df[feature_columns]
    y = df['label']
    
    print(f"Found {len(feature_columns)} feature columns.")
    print(f"Dataset shape: {X.shape}, Labels shape: {y.shape}")

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=args.test_size, 
        random_state=args.seed,
        stratify=y # Ensure same class distribution in train/test
    )
    print(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}")

    # Get and train the model
    print(f"Training {args.model} model...")
    model = get_model(args.model)
    model.fit(X_train, y_train)
    print("Training complete.")

    # Evaluate the model on the validation set
    print("\n--- Validation Results ---")
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    print(f"Validation Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_val, y_pred, target_names=['FAKE', 'REAL']))
    
    # Save the trained model along with the feature names
    print(f"\nSaving model and feature names to {args.out_model}...")
    
    saved_object = {
        'model': model,
        'feature_names': X_train.columns.tolist()
    }
    joblib.dump(saved_object, args.out_model)
    print("Model saved successfully.")

if __name__ == '__main__':
    main()
