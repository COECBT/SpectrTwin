import os
import glob
import struct
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import qmc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# --- Settings ---
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.grid': False,  
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

DATA_DIR = r"n:\Projects\Y= MX+C 2\my_app\Data\Experimental design verification"
Y_FILE = os.path.join(DATA_DIR, "y.xlsx")
SPA_DIR = os.path.join(DATA_DIR, "Data_spa")

def select_lhs_subset(y_source, n_samples):
    if n_samples >= len(y_source):
        return np.arange(len(y_source))
        
    sampler = qmc.LatinHypercube(d=1, seed=42)
    lhs_01 = sampler.random(n=n_samples)
    lhs_scaled = y_source.min() + lhs_01.flatten() * (y_source.max() - y_source.min())
    
    used_indices = set()
    subset_idx = []
    
    for target in lhs_scaled:
        distances = np.abs(y_source - target)
        sorted_closest = np.argsort(distances)
        for idx in sorted_closest:
            if idx not in used_indices:
                used_indices.add(idx)
                subset_idx.append(idx)
                break
                
    return np.array(subset_idx)

combined_excel_path = os.path.join(DATA_DIR, "combined_data.xlsx")
print("1. Loading combined data...")
combined_df = pd.read_excel(combined_excel_path)

X_raw = combined_df.drop(columns=['file', 'conc']).values
X = X_raw[:, ::5]

mean_X = np.mean(X, axis=1, keepdims=True)
std_X = np.std(X, axis=1, keepdims=True)
X = (X - mean_X) / std_X
y = combined_df['conc'].values

print("2. Initiating LHS Experimental Design Progression...")
X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

fractions = np.linspace(0.1, 1.0, 10)
results = []
models_data = {}
all_plot_points = []  # Added to store the exact R2 scatter plot points

for frac in fractions:
    n_samples = max(10, int(len(y_pool) * frac))
    print(f"Training on {n_samples} items")
    
    idx = select_lhs_subset(y_pool, n_samples)
    
    X_train = X_pool[idx]
    y_train = y_pool[idx]
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    pred_train = rf.predict(X_train)
    pred_test = rf.predict(X_test)
    
    r2_train = r2_score(y_train, pred_train)
    r2_test = r2_score(y_test, pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
    
    # Store aggregate metric results
    results.append({
        'Fraction': frac * 100,
        'Samples_Used': n_samples,
        'Train_R2': r2_train,
        'Test_R2': r2_test,
        'Train_RMSE': rmse_train,
        'Test_RMSE': rmse_test
    })
    
    # Store raw values for scatter plots tracking
    for r, p in zip(y_train, pred_train):
        all_plot_points.append({'Fraction_Used': int(frac*100), 'Samples_Used': n_samples, 'Set': 'Train', 'Reference': r, 'Predicted': p})
    for r, p in zip(y_test, pred_test):
        all_plot_points.append({'Fraction_Used': int(frac*100), 'Samples_Used': n_samples, 'Set': 'Test', 'Reference': r, 'Predicted': p})
    
    models_data[round(frac*100)] = {
        'samples': n_samples,
        'y_train': y_train, 'pred_train': pred_train, 'r2_tr': r2_train, 'rm_tr': rmse_train,
        'y_test': y_test, 'pred_test': pred_test, 'r2_te': r2_test, 'rm_te': rmse_test
    }

res_df = pd.DataFrame(results)
metrics_excel_path = os.path.join(DATA_DIR, "Performance_Metrics.xlsx")
res_df.to_excel(metrics_excel_path, index=False)
print(f"Metrics saved to {metrics_excel_path}")

# Export the raw scatter plot points data
scatter_df = pd.DataFrame(all_plot_points)
scatter_excel_path = os.path.join(DATA_DIR, "Raw_Scatter_Values.xlsx")
scatter_df.to_excel(scatter_excel_path, index=False)
print(f"Raw scatter plot values saved to {scatter_excel_path}")

print("3. Generating Plots...")

# --- Plot 1: Learning Curve ---
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(res_df['Samples_Used'], res_df['Train_R2'], marker='o', linewidth=2, color='#1f77b4', label='Train R²')
ax.plot(res_df['Samples_Used'], res_df['Test_R2'], marker='D', linewidth=2, color='#2ca02c', label='Test R²')
ax.axvspan(res_df['Samples_Used'].iloc[4], res_df['Samples_Used'].iloc[6], color='grey', alpha=0.15, label='Target Optimization Zone')
ax.set_xlabel('Number of Discovery Samples Used (LHS Design)')
ax.set_ylabel('Algorithm R² Performance')
ax.set_ylim(-0.1, 1.0)
ax.legend(loc='lower right', frameon=False)
ax.grid(False) 
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
curve_path = os.path.join(DATA_DIR, "Learning_Curve.png")
plt.savefig(curve_path, dpi=600)
plt.close()

# --- Plot 2: Scatter Grids for ALL Cases ---
fig, axes = plt.subplots(2, 5, figsize=(25, 10))
axes = axes.flatten()

for i, frac in enumerate(range(10, 110, 10)):
    ax = axes[i]
    data = models_data[frac]
    ax.scatter(data['y_train'], data['pred_train'], c='#1f77b4', edgecolors='None', alpha=0.6, label=f"Train R² = {data['r2_tr']:.2f}")
    ax.scatter(data['y_test'], data['pred_test'], c='#d62728', edgecolors='None', alpha=0.6, marker='s', label=f"Test R² = {data['r2_te']:.2f}")
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=1.5)
    
    # Internal Text identifier replacing generic percentages with Sample counts!
    ax.text(0.5, 0.1, f"{data['samples']} Samples", transform=ax.transAxes, ha='center', va='center', weight='bold', size=14)
    
    ax.set_xlabel('Reference Concentration' if i >= 5 else "")
    ax.set_ylabel('Predicted' if i % 5 == 0 else "")
    
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', prop={'size': 10}, frameon=False)

plt.tight_layout()
scat_path = os.path.join(DATA_DIR, "R2_Comparisons.png")
plt.savefig(scat_path, dpi=600)
plt.close()

# --- Plot 3: 3-Case Explainable Normalized Spider Plot ---
spider_labels = [
    'Train Confidence\n(Train R²)', 
    'Interpolation Strength\n(Test R²)', 
    'Physical Accuracy\n(Normalized Train RMSE)', 
    'Prediction Scaling\n(Normalized Test RMSE)'
]

max_rm_tr = res_df['Train_RMSE'].max()
max_rm_te = res_df['Test_RMSE'].max()

stats = {}
for frac in [10, 50, 100]:
    d = models_data[frac]
    # R2 is inherently 0-1.
    # We normalize RMSE so higher visually = lower physical error (better)
    norm_rm_tr = max(0, 1.0 - (d['rm_tr'] / max_rm_tr))
    norm_rm_te = max(0, 1.0 - (d['rm_te'] / max_rm_te))
    
    s = [max(0.01, d['r2_tr']), max(0.01, d['r2_te']), max(0.01, norm_rm_tr), max(0.01, norm_rm_te)]
    s += s[:1]
    stats[frac] = s

angles = np.linspace(0, 2*np.pi, 4, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), spider_labels, weight='bold')

ax.grid(False)
ax.set_ylim(0, 1.0)
ax.set_yticks([]) 
ax.spines['polar'].set_visible(False)

colors = {10: '#d62728', 50: '#ff7f0e', 100: '#1f77b4'}
names = {
    10: f"{models_data[10]['samples']} Samples", 
    50: f"{models_data[50]['samples']} Samples\n(Saturation Point)", 
    100: f"{models_data[100]['samples']} Samples\n(Total Full Data)"
}

for frac in [10, 50, 100]:
    ax.plot(angles, stats[frac], color=colors[frac], linewidth=3, linestyle='solid', marker='o', label=names[frac])
    ax.fill(angles, stats[frac], color=colors[frac], alpha=0.1)

# Annotate logic explaining the plot
plt.figtext(0.5, 0.05, "*All metrics mapped scale outwards. Visual maximum area represents standard absolute optimization.", 
            ha="center", fontsize=10, style='italic', color='gray')

plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3, frameon=False)
spider_path = os.path.join(DATA_DIR, "Performance_Spider_Plot.png")
plt.savefig(spider_path, dpi=600, bbox_inches='tight')
plt.close()

print("Saved R2 Grids, Excel Metrics, and Spider Plot")
print("Analysis Complete!")
