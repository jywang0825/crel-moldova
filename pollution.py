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
gdp_file   = glob.glob('/Users/joseph/Downloads/gdp_quarterly*.csv')[0]
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
# 3. Regression: GDP ~ each pollutant (spatial avg) individually + plot
# ------------------------------------------------------------------------------
print("=== Regression: GDP ~ each pollutant (spatial avg) individually ===")
for var in pollutant_vars:
    X = df[[var]]
    lr = LinearRegression().fit(X, y)
    r2 = lr.score(X, y)
    print(f"{var:10} → Intercept: {lr.intercept_:.4f},  Slope: {lr.coef_[0]:.4f},  R²: {r2:.4f}")
    # scatter + regression line
    plt.figure(figsize=(5,4))
    plt.scatter(df[var], y, alpha=0.7, label='data')
    xs = np.linspace(df[var].min(), df[var].max(), 100).reshape(-1,1)
    plt.plot(xs, lr.predict(xs), '--', label=f'fit (R²={r2:.2f})')
    plt.xlabel(f'{var} (spatial avg)')
    plt.ylabel('GDP')
    plt.title(f'GDP vs {var}')
    plt.legend()
    plt.tight_layout()
    plt.show()

print("\n=== Combined Regression: GDP ~ all pollutant spatial averages ===")
X_all = df[pollutant_vars]
lr_all = LinearRegression().fit(X_all, y)
print(f"Intercept: {lr_all.intercept_:.4f}")
for name, coef in zip(pollutant_vars, lr_all.coef_):
    print(f"  {name:>10}: {coef:.4f}")
print(f"R²:        {lr_all.score(X_all, y):.4f}\n")

# scatter + fit for combined multivariate
plt.figure(figsize=(6,4))
# project multivariate to 1D via its own regression predictions
y_pred_all = lr_all.predict(X_all)
plt.scatter(y_pred_all, y, alpha=0.7, label='data')
lr_line_all = LinearRegression().fit(y_pred_all.reshape(-1,1), y)
xs_all = np.linspace(y_pred_all.min(), y_pred_all.max(), 100)
plt.plot(xs_all, lr_line_all.predict(xs_all.reshape(-1,1)), '--',
         label=f'fit (slope={lr_line_all.coef_[0]:.2f})')
plt.xlabel('Predicted GDP (pollutant avg model)')
plt.ylabel('Actual GDP')
plt.title('Actual vs Predicted GDP\nPollutant-avg Multivariate Model')
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# 4. PCA on [one pollutant + trends] → keep PCs ≥ 90% → combined regression + plot
# ------------------------------------------------------------------------------
chosen_poll = pollutant_vars[0]
combined_cols = [chosen_poll] + trend_cols

pca_mix       = PCA().fit(df[combined_cols])
explained_mix = pca_mix.explained_variance_ratio_
cumvar_mix    = np.cumsum(explained_mix)
k_mix         = np.searchsorted(cumvar_mix, 0.90) + 1

print(f"PCA on [{chosen_poll} + trends] explained variance:")
for i, ev in enumerate(explained_mix, start=1):
    print(f"  PC{i}: {ev*100:5.2f}%")
print(f"\nRetaining {k_mix} PCs (≥ 90% variance)\n")

mix_pc_cols = [f'Mix_PC{i}' for i in range(1, k_mix+1)]
df[mix_pc_cols] = pca_mix.transform(df[combined_cols])[:, :k_mix]

X_mix_all  = df[mix_pc_cols]
lr_mix_all = LinearRegression().fit(X_mix_all, y)
r2_mix_all = lr_mix_all.score(X_mix_all, y)

print(f"=== Combined Regression: GDP ~ ({chosen_poll} + trends) PCs ===")
print(f"Intercept: {lr_mix_all.intercept_:.4f}")
for name, coef in zip(mix_pc_cols, lr_mix_all.coef_):
    print(f"  {name:>8}: {coef:.4f}")
print(f"R²:        {r2_mix_all:.4f}\n")

# plot Actual GDP vs combined-mix PCA prediction
y_pred_mix = lr_mix_all.predict(X_mix_all)
plt.figure(figsize=(6,6))
plt.scatter(y_pred_mix, y, alpha=0.7, label='data')
lr_line = LinearRegression().fit(y_pred_mix.reshape(-1,1), y)
xs = np.linspace(y_pred_mix.min(), y_pred_mix.max(), 100)
plt.plot(xs, lr_line.predict(xs.reshape(-1,1)), '--',
         label=f'fit (slope={lr_line.coef_[0]:.3f})')
plt.xlabel('Combined Mix PCA Score')
plt.ylabel('Actual GDP')
plt.title(f'Actual GDP vs {chosen_poll}+Trends PCA Output')
plt.legend()
plt.tight_layout()
plt.show()
