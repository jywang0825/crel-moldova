import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

def run_pca(file_path, target_col_index=0, date_column_name='date'):
    # --- Load Data ---

    df = pd.read_csv(file_path, skiprows=[1], thousands=',')
    
    # --- Drop Date Column ---
    date_cols = [col for col in df.columns if col.lower() == date_column_name.lower()]
    if date_cols:
        df = df.drop(columns=date_cols)
    
    # --- Convert All Columns to Numeric ---
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # --- Identify and Report Columns with Missing Values ---
    nan_counts = df.isna().sum()
    problematic_columns = nan_counts[nan_counts > 0]
    if not problematic_columns.empty:
        print(f"Warning: Found missing values (NaN) in file: {file_path}")
        print(problematic_columns)
        # Optionally, you may impute or drop these values.
    else:
        print(f"No missing values found in {file_path}.")
    
    # --- Prepare the Data for PCA ---
    X = df.values
    
    # Standardize the data before applying PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # --- Perform PCA ---
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    # --- Explained Variance and Loadings ---
    explained_variance = pca.explained_variance_ratio_
    print(f"\nExplained Variance Ratios for {file_path}:")
    for i, var in enumerate(explained_variance):
        print(f"  Component {i+1}: {var * 100:.2f}%")
    
    loadings = pca.components_
    print("\nComponent Loadings (rows = components, columns = original features):")
    print(loadings)
    
    # --- Reconstruction for Predicted vs. Actual ---
    X_reconstructed = pca.inverse_transform(X_pca)
    
    # For demonstration, choose a target column (default: first column)
    actual = X_scaled[:, target_col_index]
    predicted = X_reconstructed[:, target_col_index]
    
    # --- Plot Predicted vs. Actual ---
    plt.figure(figsize=(8, 6))
    plt.scatter(actual, predicted, alpha=0.7)
    # Plot the ideal line: y = x
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], color='red', lw=2)
    plt.xlabel("Actual (Standardized)")
    plt.ylabel("Predicted (Standardized)")
    plt.title(f"Predicted vs. Actual for Column Index {target_col_index}\nFile: {file_path}")
    plt.grid(True)
    plt.show()
    
    # --- Compute Performance Metrics ---
    r2 = r2_score(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    print(f"\nMetrics for {file_path} (Column Index {target_col_index}):")
    print(f"  R^2 Score: {r2:.4f}")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    
    # Return all results in a dictionary for later use if needed.
    return {
        'explained_variance': explained_variance,
        'loadings': loadings,
        'r2': r2,
        'mse': mse,
        'pca_object': pca
    }

# --- Example Usage for Multiple Files ---
file_list = [
    "/Users/joseph/Downloads/Block1.csv",
    "/Users/joseph/Downloads/Block2.csv",
    "/Users/joseph/Downloads/Block3.csv",
    "/Users/joseph/Downloads/Block4.csv"
]

results_dict = {}
for file_path in file_list:
    print("\n" + "="*60)
    print(f"Processing file: {file_path}")
    results_dict[file_path] = run_pca(file_path)
    print("="*60)
