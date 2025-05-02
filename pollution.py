import pandas as pd
import numpy as np
import xarray as xr
import glob
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# 1. Load & spatially AVERAGE each pollutant → daily → quarterly
# ------------------------------------------------------------------------------
ds = xr.open_dataset('/Users/joseph/Downloads/dataset.nc')
pollutant_vars = list(ds.data_vars)

start = pd.to_datetime('2021-07-01')
mean_ts = {}
for var in pollutant_vars:
    ts = ds[var].mean(dim=['level','latitude','longitude']).to_series()
    ts.index = start + pd.to_timedelta(ts.index, unit='D')
    mean_ts[var] = ts

df_poll = pd.DataFrame(mean_ts).resample('Q').mean().sort_index()

# ------------------------------------------------------------------------------
# 2. Load GDP & Trends → align lengths
# ------------------------------------------------------------------------------
gdp_file   = glob.glob('/Users/joseph/Downloads/gdp_quarterly.csv')[0]
gdp_vals   = pd.read_csv(gdp_file).iloc[:, 1].values

trends_df  = pd.read_csv('/Users/joseph/Downloads/quarterly_trends_averages.csv')
trend_cols = trends_df.columns[1:].tolist()
trend_vals = trends_df[trend_cols].values

n = min(len(df_poll), len(gdp_vals), trend_vals.shape[0])
df = df_poll.iloc[:n].copy()
df['gdp'] = gdp_vals[:n]
for i, col in enumerate(trend_cols):
    df[col] = trend_vals[:n, i]

y = df['gdp']

# ------------------------------------------------------------------------------
# 3. Regression: GDP ~ pollutant spatial AVERAGE (no PCA)
# ------------------------------------------------------------------------------
print("=== Regression 3: GDP ~ each pollutant (spatial avg) individually ===")
for var in pollutant_vars:
    X = df[[var]]
    lr = LinearRegression().fit(X, y)
    r2 = lr.score(X, y)
    print(f"{var:10} → Intercept: {lr.intercept_:.4f},  Slope: {lr.coef_[0]:.4f},  R²: {r2:.4f}")
    # optional per‐pollutant scatter+fit
    plt.figure(figsize=(5,4))
    plt.scatter(df[var], y, alpha=0.7)
    xs = np.linspace(df[var].min(), df[var].max(), 100).reshape(-1,1)
    plt.plot(xs, lr.predict(xs), '--')
    plt.xlabel(f'{var} (spatial avg)')
    plt.ylabel('GDP')
    plt.title(f'GDP vs {var}')
    plt.tight_layout()
    plt.show()

print("\n=== Combined Regression: GDP ~ all pollutant spatial averages ===")
X_all = df[pollutant_vars]
lr_all = LinearRegression().fit(X_all, y)
print(f"Intercept: {lr_all.intercept_:.4f}")
for name, coef in zip(pollutant_vars, lr_all.coef_):
    print(f"  {name:>10}: {coef:.4f}")
print(f"R²:        {lr_all.score(X_all, y):.4f}\n")

# ------------------------------------------------------------------------------
# 4. Combined PCA on [pollution‐avg + trends] → regressions & plots
# ------------------------------------------------------------------------------
combined_cols   = pollutant_vars + trend_cols
pca_comb        = PCA().fit(df[combined_cols])
explained_comb  = pca_comb.explained_variance_ratio_
cumvar_comb     = np.cumsum(explained_comb)
k_comb          = np.searchsorted(cumvar_comb, 0.90) + 1

print("Combined PCA explained variance:")
for i, ev in enumerate(explained_comb, start=1):
    print(f"  PC{i}: {ev*100:5.2f}%")
print(f"\nRetaining {k_comb} PCs (≥ 90% variance)\n")

comb_pc_cols = [f'Comb_PC{i}' for i in range(1, k_comb+1)]
df[comb_pc_cols] = pca_comb.transform(df[combined_cols])[:, :k_comb]

# single multivariate regression on all retained Combined PCs
X_comb_all  = df[comb_pc_cols]
lr_comb_all = LinearRegression().fit(X_comb_all, y)
print("=== Combined Regression: GDP ~ (Pollution Avg + Trends) PCs ===")
print(f"Intercept: {lr_comb_all.intercept_:.4f}")
for name, coef in zip(comb_pc_cols, lr_comb_all.coef_):
    print(f"  {name:>8}: {coef:.4f}")
print(f"R²:        {lr_comb_all.score(X_comb_all, y):.4f}\n")

# Plot: Actual vs Predicted GDP for the combined‐PCA model
y_pred_comb = lr_comb_all.predict(X_comb_all)
plt.figure(figsize=(6,6))
plt.scatter(y_pred_comb, y, alpha=0.7, label='data')
lr_line = LinearRegression().fit(y_pred_comb.reshape(-1,1), y)
xs = np.linspace(y_pred_comb.min(), y_pred_comb.max(), 100)
plt.plot(xs, lr_line.predict(xs.reshape(-1,1)), '--', label=f'fit: slope={lr_line.coef_[0]:.3f}')
plt.xlabel('Combined PCA Model Output')
plt.ylabel('Actual GDP')
plt.title('Actual GDP vs Combined PCA Model Output')
plt.legend()
plt.tight_layout()
plt.show()
