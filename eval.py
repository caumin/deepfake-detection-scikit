import pandas as pd
import argparse
import joblib
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.base import clone
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from features import get_feature_names

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a test dataset.")
    parser.add_argument('--csv', type=str, required=True, help="Path to the test features CSV file.")
    parser.add_argument('--model', type=str, required=True, help="Path to the trained model (.joblib file).")
    parser.add_argument('--report_dir', type=str, default='out/report', help="Directory to save evaluation report and plots.")
    parser.add_argument('--feature-importance', action='store_true', help="If set, generate and save a feature importance plot for tree-based models.")
    parser.add_argument('--plot-decision-boundary', action='store_true', help="If set, generate and save a 2D decision boundary plot.")
    parser.add_argument('--train-csv', type=str, help="Path to the training features CSV file, required for decision boundary plot.")
    
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.report_dir, exist_ok=True)

    # --- Load Model & Training Feature Names ---
    print(f"Loading model and feature names from {args.model}...")
    try:
        saved_object = joblib.load(args.model)
        model = saved_object['model']
        training_feature_names = saved_object['feature_names']
    except Exception as e:
        print(f"Error loading model or feature names: {e}")
        print("Ensure the model file was saved with both the model and 'feature_names'.")
        return

    # --- Load Data & Align Columns ---
    print(f"Loading test data from {args.csv}...")
    df = pd.read_csv(args.csv)

    # Use the feature names from the trained model to align the test data
    try:
        # Ensure all required columns are present
        missing_cols = set(training_feature_names) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in the test data: {', '.join(missing_cols)}")

        X_test = df[training_feature_names] # Select and order columns
        y_test = df['label']
    except Exception as e:
        print(f"Error preparing data: {e}")
        return
        
    print(f"Found {len(X_test)} samples in the test set.")

    # --- Evaluation ---
    print("Evaluating model performance...")
    y_pred = model.predict(X_test)
    
    # Calculate probabilities for AUROC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1] # Probability of the 'REAL' class
        auroc = roc_auc_score(y_test, y_prob)
    else: # Models like SVC with linear kernel might not have predict_proba by default
        y_prob = None
        auroc = "N/A (model does not support predict_proba)"

    accuracy = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, target_names=['FAKE', 'REAL'], output_dict=True)
    
    print("\n--- Test Set Evaluation Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUROC: {auroc if isinstance(auroc, str) else f'{auroc:.4f}'}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['FAKE', 'REAL']))

    # --- Save Report ---
    report = {
        'model_path': args.model,
        'test_data_path': args.csv,
        'accuracy': accuracy,
        'auroc': auroc,
        'classification_report': report_dict
    }

    report_path = os.path.join(args.report_dir, 'report.json')
    print(f"\nSaving detailed report to {report_path}...")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    print("Report saved.")

    # --- Save Confusion Matrix Plot ---
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['FAKE', 'REAL'])
    
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap='Blues')
    plt.title(f'Confusion Matrix for {os.path.basename(args.model)}')
    
    cm_path = os.path.join(args.report_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix plot saved to {cm_path}")
    plt.close()

    # --- Feature Importance Plot ---
    if args.feature_importance:
        # To handle pipelines, get the base model first
        if isinstance(model, Pipeline):
            base_model = model.steps[-1][1]
        else:
            base_model = model

        if hasattr(base_model, 'feature_importances_'):
            print("Generating feature importance plot...")
            
            # Get feature importances
            importances = base_model.feature_importances_
            feature_names = X_test.columns
            
            # Create a DataFrame for better handling
            feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
            
            # Sort by importance and select top 20
            top_features = feature_importance_df.sort_values(by='importance', ascending=False).head(20)
            
            # Plot
            plt.figure(figsize=(12, 10))
            plt.barh(top_features['feature'], top_features['importance'], color='skyblue')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title(f'Top 20 Feature Importances for {os.path.basename(args.model)}')
            plt.gca().invert_yaxis()  # Display the most important feature at the top
            plt.tight_layout()
            
            # Save the plot
            importance_plot_path = os.path.join(args.report_dir, 'feature_importance.png')
            plt.savefig(importance_plot_path)
            print(f"Feature importance plot saved to {importance_plot_path}")
            plt.close()
        else:
            print("Model does not have 'feature_importances_' attribute. Skipping plot.")

    # --- Decision Boundary Plot ---
    if args.plot_decision_boundary:
        if not args.train_csv:
            print("\n--plot-decision-boundary requires --train-csv. Skipping plot.")
            return

        print("\nGenerating decision boundary plot...")
        try:
            # 1. Load training data for fitting PCA
            train_df = pd.read_csv(args.train_csv)
            # Align columns just in case
            X_train_full = train_df[training_feature_names]
            y_train_full = train_df['label']

            # 2. Fit PCA on training data and transform
            pca = PCA(n_components=2, random_state=42)
            X_train_2d = pca.fit_transform(X_train_full)
            X_test_2d = pca.transform(X_test)

            # 3. Train a new classifier on the 2D data
            # Clone the original model's classifier step to keep its parameters.
            # Handle both raw estimators and pipelines.
            if isinstance(model, Pipeline):
                classifier_2d = clone(model.steps[-1][1])
            else:
                classifier_2d = clone(model)
            
            # The model inside the pipeline might have n_jobs that can't be pickled, handle this
            if hasattr(classifier_2d, 'n_jobs'):
                classifier_2d.n_jobs = 1 
            classifier_2d.fit(X_train_2d, y_train_full)

            # 4. Create meshgrid for plotting
            x_min, x_max = X_test_2d[:, 0].min() - 1, X_test_2d[:, 0].max() + 1
            y_min, y_max = X_test_2d[:, 1].min() - 1, X_test_2d[:, 1].max() + 1
            # Use np.linspace to create a grid of fixed resolution (e.g., 400x400)
            # This is robust to the scale of the data and avoids MemoryError.
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                                 np.linspace(y_min, y_max, 400))

            # 5. Predict on meshgrid
            Z = classifier_2d.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            # 6. Plot
            plt.figure(figsize=(12, 10))
            plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.5)

            # Overlay test data points
            scatter = plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test, cmap=plt.cm.RdYlBu,
                                  edgecolor='k', s=25)
            
            plt.title(f'2D Decision Boundary for {os.path.basename(args.model)}\n(PCA-reduced data)')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            
            # Create a legend
            handles, _ = scatter.legend_elements()
            # Ensure labels match the actual classes present in y_test
            class_labels = ['FAKE', 'REAL']
            legend_labels = [class_labels[i] for i in sorted(y_test.unique())]
            plt.legend(handles, legend_labels, title="Classes")

            # Save the plot
            boundary_plot_path = os.path.join(args.report_dir, 'decision_boundary.png')
            plt.savefig(boundary_plot_path)
            print(f"Decision boundary plot saved to {boundary_plot_path}")
            plt.close()

        except Exception as e:
            print(f"Could not generate decision boundary plot. Error: {e}")
            
    # --- SHAP Summary Plot ---
    print("\nGenerating SHAP summary plot...")
    try:
        # Get the base model, handling both pipelines and raw estimators
        if isinstance(model, Pipeline):
            base_model = model.steps[-1][1]
        else:
            base_model = model
        
        # SHAP works best with specific explainers, here we support tree models
        if isinstance(base_model, (XGBClassifier, RandomForestClassifier)):
            # For large datasets, SHAP can be slow. Subsample the data for the explainer.
            X_test_shap = X_test
            if len(X_test) > 2000:
                print(f"Subsampling {len(X_test)} test samples to 2000 for SHAP calculation.")
                X_test_shap = X_test.sample(n=2000, random_state=42)

            explainer = shap.TreeExplainer(base_model)
            shap_values = explainer.shap_values(X_test_shap)
            
            # The output of shap_values can be a list (for multi-class) or a single array
            # For binary classification, it's often a list of two arrays. We take the one for the "REAL" class (class 1)
            shap_values_for_plot = shap_values[1] if isinstance(shap_values, list) else shap_values

            # Generate and save the plot
            shap.summary_plot(shap_values_for_plot, X_test_shap, show=False)
            
            plt.title(f'SHAP Summary for {os.path.basename(args.model)}')
            shap_plot_path = os.path.join(args.report_dir, 'shap_summary.png')
            plt.savefig(shap_plot_path, bbox_inches='tight')
            print(f"SHAP summary plot saved to {shap_plot_path}")
            plt.close()
        else:
            print(f"SHAP plot generation not supported for model type: {type(base_model).__name__}. Skipping.")

    except Exception as e:
        print(f"Could not generate SHAP plot. Error: {e}")

    # --- Permutation Importance Plot ---
    print("\nCalculating permutation importance...")
    try:
        result = permutation_importance(
            model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
        )
        
        # Create a DataFrame for easier handling
        perm_importance_df = pd.DataFrame({
            'feature': X_test.columns,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std,
        })
        
        # Sort by importance and select top 20
        top_features = perm_importance_df.sort_values(
            by='importance_mean', ascending=False
        ).head(20)

        # Plot
        plt.figure(figsize=(12, 10))
        plt.barh(
            top_features['feature'],
            top_features['importance_mean'],
            xerr=top_features['importance_std'],
            align='center',
            color='lightgreen',
            ecolor='gray',
        )
        plt.xlabel('Permutation Importance (mean)')
        plt.ylabel('Feature')
        plt.title(f'Top 20 Permutation Importances for {os.path.basename(args.model)}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        # Save the plot
        perm_importance_plot_path = os.path.join(args.report_dir, 'permutation_importance.png')
        plt.savefig(perm_importance_plot_path)
        print(f"Permutation importance plot saved to {perm_importance_plot_path}")
        plt.close()

    except Exception as e:
        print(f"Could not calculate permutation importance. Error: {e}")


if __name__ == '__main__':
    main()
