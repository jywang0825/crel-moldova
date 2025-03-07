import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the dataset from the Excel file
df = pd.read_excel("/Users/joseph/Downloads/quarterly_trends_averages.xlsx")

# Select only numeric columns for PCA (if your dataset has non-numeric columns)
numeric_df = df.select_dtypes(include=[np.number])
print("\nNumeric data used for PCA:")
print(numeric_df.head())

# Convert the DataFrame to a NumPy array
X = numeric_df.values

# Initialize PCA using n_components=0.9 (choose the minimum number of components that explain 90% of the variance)
pca = PCA(n_components=0.9)
X_pca = pca.fit_transform(X)

# Print PCA results
print("Number of components chosen to explain 90% variance:", pca.n_components_)
print("Explained variance ratio of the chosen components:", pca.explained_variance_ratio_)
print("Total variance explained:", pca.explained_variance_ratio_.sum())
print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)
print("\nPrincipal Components (each row corresponds to a component):")
print(pca.components_)
print("\nPCA-transformed data:")
print(X_pca)

# Print which two principal components (PC1 and PC2) are used for the plots,
# along with the percentage of variance they explain.
print("\nUsing the following two principal components for plotting:")
for i in range(min(2, pca.n_components_)):
    variance_percent = pca.explained_variance_ratio_[i] * 100
    print(f"Principal Component {i+1} (PC{i+1}) explains {variance_percent:.2f}% of the variance.")

# Plot 1: Scatter Plot of PCA-transformed Data (using first two components)
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], edgecolor='k', s=50)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Scatter Plot')
plt.grid(True)
plt.show()

# Plot 2: Explained Variance Ratio (Scree Plot)
plt.figure(figsize=(8, 6))
components = range(1, len(pca.explained_variance_ratio_) + 1)
plt.bar(components, pca.explained_variance_ratio_, edgecolor='k')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Principal Components')
plt.xticks(components)
plt.show()

# Optional Plot 3: Biplot to show loadings along with the scatter plot
def biplot(score, coeff, labels=None):
    xs = score[:, 0]
    ys = score[:, 1]
    # Scale the coefficients for better visualization
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    
    plt.figure(figsize=(8, 6))
    plt.scatter(xs * scalex, ys * scaley, edgecolor='k', s=50, label='Samples')
    
    # Plot arrows for each feature loading
    for i in range(coeff.shape[0]):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', width=0.005, head_width=0.05)
        if labels is None:
            plt.text(coeff[i, 0]*1.15, coeff[i, 1]*1.15, f"Var{i+1}", color='r')
        else:
            plt.text(coeff[i, 0]*1.15, coeff[i, 1]*1.15, labels[i], color='r')
    
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Biplot")
    plt.grid(True)
    plt.legend()
    plt.show()

# Create a biplot to visualize how each original feature contributes
biplot(X_pca, pca.components_.T, labels=numeric_df.columns)

# Additional Plot: Side-by-Side Comparison of "Before" and "After" PCA
# For the "before" plot, we'll use the first two numeric columns (if available)
if numeric_df.shape[1] >= 2:
    original_data = numeric_df.iloc[:, :2].values
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Before PCA: Original data (first two features)
    ax[0].scatter(original_data[:, 0], original_data[:, 1], edgecolor='k', s=50)
    ax[0].set_title("Before PCA (First 2 Numeric Features)")
    ax[0].set_xlabel("Feature 1")
    ax[0].set_ylabel("Feature 2")
    ax[0].grid(True)
    
    # After PCA: PCA-transformed data (first two principal components)
    ax[1].scatter(X_pca[:, 0], X_pca[:, 1], edgecolor='k', s=50)
    ax[1].set_title("After PCA (2 Principal Components)")
    ax[1].set_xlabel("Principal Component 1")
    ax[1].set_ylabel("Principal Component 2")
    ax[1].grid(True)
    
    plt.tight_layout()
    plt.show()
else:
    print("Not enough numeric features to display a before vs. after plot.")

print("Full DataFrame shape:", df.shape)
print("Numeric DataFrame shape:", numeric_df.shape)
