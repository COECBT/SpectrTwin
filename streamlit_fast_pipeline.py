"""
Fast Streamlit Pipeline with Comprehensive Timing & Validation
Runs all preprocessing, key feature selections, and essential models with timing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import time
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pathlib import Path
from scipy.signal import savgol_filter

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.grid'] = False

class FastStreamlitPipeline:
    def __init__(self):
        self.output_dir = "STREAMLIT_FAST_RESULTS"
        Path(self.output_dir).mkdir(exist_ok=True)
        self.timing_data = []
        self.all_results = {}
    
    def load_data(self):
        """Load data"""
        start = time.time()
        data = pd.read_csv("Data/Multi-objective verification/Data_raman.csv")
        X = data.drop(columns=['glucose', 'Na_acetate', 'Mg_SO4']).values
        y = data[['glucose', 'Na_acetate', 'Mg_SO4']].values
        elapsed = time.time() - start
        self.timing_data.append(('Data Loading', elapsed))
        print(f"✓ Data loaded: {X.shape} in {elapsed:.3f}s")
        return X, y
    
    def process_target(self, X, y, target_name):
        """Process single target"""
        print(f"\n{'='*70}\n{target_name.upper()}\n{'='*70}")
        results = []
        
        # Split
        start = time.time()
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        t1 = time.time() - start
        self.timing_data.append((f'Split_{target_name}', t1))
        print(f"✓ Split: {t1:.3f}s")
        
        # Preprocessing
        prep_times = {}
        preps = {
            'None': lambda x: x,
            'StandardScaler': lambda x: StandardScaler().fit_transform(x),
            'MinMax': lambda x: MinMaxScaler().fit_transform(x),
            'SNV': lambda x: (x - np.mean(x, axis=1, keepdims=True)) / (np.std(x, axis=1, keepdims=True) + 1e-10),
            'BaselineCorrection': self._baseline,
        }
        
        print("\nPreprocessing:")
        processed = {}
        for name, func in preps.items():
            start = time.time()
            processed[name] = (func(X_train), func(X_val), func(X_test))
            t = time.time() - start
            prep_times[name] = t
            self.timing_data.append((f'Prep_{name}_{target_name}', t))
            print(f"  ✓ {name:20s} {t:.3f}s")
        
        # Feature Selection
        print("\nFeature Selection:")
        features = {}
        for k in [100, 200]:
            start = time.time()
            sel = SelectKBest(f_regression, k=min(k, X_train.shape[1]))
            X_tr_s = sel.fit_transform(X_train, y_train)
            X_v_s, X_te_s = sel.transform(X_val), sel.transform(X_test)
            t = time.time() - start
            features[f'k{k}'] = (X_tr_s, X_v_s, X_te_s)
            self.timing_data.append((f'FeatSel_k{k}_{target_name}', t))
            print(f"  ✓ k={k} {t:.3f}s")
        
        # Models Training
        print("\nModel Training:")
        models_def = {
            'Ridge': Ridge(alpha=1),
            'Lasso': Lasso(alpha=0.001),
            'ElasticNet': ElasticNet(),
            'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(random_state=42),
        }
        
        start_all = time.time()
        for prep_name, (Xtr, Xv, Xte) in processed.items():
            sel_name = 'k100'
            sel = SelectKBest(f_regression, k=100)
            Xtr_s = sel.fit_transform(Xtr, y_train)
            Xv_s, Xte_s = sel.transform(Xv), sel.transform(Xte)
            
            for model_name, model in models_def.items():
                model.fit(Xtr_s, y_train)
                r2 = r2_score(y_test, model.predict(Xte_s))
                if r2 > 0.5:
                    print(f"  ✓ {prep_name:15s} + {model_name:15s} R²={r2:.3f}")
                results.append({
                    'preprocessing': prep_name,
                    'features': sel_name,
                    'model': model_name,
                    'r2': r2
                })
        
        t_models = time.time() - start_all
        self.timing_data.append((f'ModelTraining_{target_name}', t_models))
        print(f"\nModel training total: {t_models:.3f}s")
        
        return pd.DataFrame(results)
    
    def _baseline(self, X):
        X_out = X.copy()
        for i in range(X.shape[0]):
            baseline = np.convolve(X[i], np.ones(10)/10, mode='same')
            X_out[i] = X[i] - baseline
        return X_out
    
    def generate_all_plots(self, results_dict):
        """Generate all visualization plots"""
        # Timing analysis
        self._plot_timing()
        
        # Results by target
        for target, df in results_dict.items():
            self._plot_results(target, df)
        
        # Combined comparison
        self._plot_combined_comparison(results_dict)
    
    def _plot_timing(self):
        """Generate timing plot"""
        print(f"\nGenerating timing plot...")
        df = pd.DataFrame(self.timing_data, columns=['Step', 'Time'])
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Sorted bars
        df_sorted = df.sort_values('Time', ascending=True)
        ax = axes[0]
        ax.barh(range(len(df_sorted)), df_sorted['Time'], color=plt.cm.Set3(np.linspace(0, 1, len(df_sorted))))
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels([s[:25] for s in df_sorted['Step']], fontsize=9)
        ax.set_xlabel('Time (seconds)', fontsize=11, weight='bold')
        ax.set_title('Execution Time per Step', fontsize=12, weight='bold')
        ax.grid(False)
        
        # Cumulative
        ax = axes[1]
        df_sorted_all = df.sort_values('Time')
        y_pos = np.arange(len(df_sorted_all))
        colors = plt.cm.viridis(y_pos / len(df_sorted_all))
        ax.scatter(df_sorted_all['Time'], y_pos, s=100, c=colors, alpha=0.6)
        ax.set_xlabel('Time (seconds)', fontsize=11, weight='bold')
        ax.set_ylabel('Step Index', fontsize=11, weight='bold')
        ax.set_title('Timing Distribution', fontsize=12, weight='bold')
        ax.grid(False)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/00_TIMING_ANALYSIS.png', dpi=600, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: 00_TIMING_ANALYSIS.png")
    
    def _plot_results(self, target, df):
        """Plot results for target"""
        print(f"Generating results plot for {target}...")
        
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        
        # Best by model
        ax = axes[0, 0]
        best_models = df.loc[df.groupby('model')['r2'].idxmax()].sort_values('r2', ascending=False)
        ax.barh(range(len(best_models)), best_models['r2'], color='steelblue')
        ax.set_yticks(range(len(best_models)))
        ax.set_yticklabels(best_models['model'])
        ax.set_xlabel('Best R²', fontsize=10, weight='bold')
        ax.set_title(f'{target}: Best Model Performance', fontsize=11, weight='bold')
        ax.grid(False)
        
        # By preprocessing
        ax = axes[0, 1]
        prep_perf = df.groupby('preprocessing')['r2'].mean().sort_values()
        ax.barh(range(len(prep_perf)), prep_perf, color='coral')
        ax.set_yticks(range(len(prep_perf)))
        ax.set_yticklabels(prep_perf.index)
        ax.set_xlabel('Mean R²', fontsize=10, weight='bold')
        ax.set_title(f'{target}: Preprocessing Effectiveness', fontsize=11, weight='bold')
        ax.grid(False)
        
        # Distribution
        ax = axes[1, 0]
        ax.hist(df['r2'], bins=15, color='skyblue', edgecolor='black')
        ax.axvline(df['r2'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {df['r2'].mean():.3f}")
        ax.axvline(df['r2'].max(), color='green', linestyle='--', linewidth=2, label=f"Max: {df['r2'].max():.3f}")
        ax.set_xlabel('R² Score', fontsize=10, weight='bold')
        ax.set_ylabel('Frequency', fontsize=10, weight='bold')
        ax.set_title(f'{target}: R² Distribution', fontsize=11, weight='bold')
        ax.legend()
        ax.grid(False)
        
        # Top configs
        ax = axes[1, 1]
        top = df.nlargest(10, 'r2').copy()
        top['config'] = top['preprocessing'] + '+' + top['model']
        top_sorted = top.sort_values('r2')
        ax.barh(range(len(top_sorted)), top_sorted['r2'], color='mediumpurple')
        ax.set_yticks(range(len(top_sorted)))
        ax.set_yticklabels([c[:20] for c in top_sorted['config']], fontsize=8)
        ax.set_xlabel('R²', fontsize=10, weight='bold')
        ax.set_title(f'{target}: Top 10 Configurations', fontsize=11, weight='bold')
        ax.grid(False)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/Results_{target}.png', dpi=600, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: Results_{target}.png")
    
    def _plot_combined_comparison(self, results_dict):
        """Compare all targets"""
        print(f"Generating comparison plot...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, (target, df) in enumerate(results_dict.items()):
            ax = axes[idx]
            
            # Best by model
            best = df.loc[df.groupby('model')['r2'].idxmax()].sort_values('r2', ascending=False).head(8)
            colors = plt.cm.Set3(np.linspace(0, 1, len(best)))
            ax.bar(range(len(best)), best['r2'], color=colors)
            ax.set_xticks(range(len(best)))
            ax.set_xticklabels(best['model'], rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('R²', fontsize=10, weight='bold')
            ax.set_title(f'{target}', fontsize=11, weight='bold')
            ax.set_ylim([0, 1])
            ax.grid(False)
            
            # Add value labels
            for i, v in enumerate(best['r2']):
                ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/ALL_TARGETS_COMPARISON.png', dpi=600, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: ALL_TARGETS_COMPARISON.png")
    
    def save_excel(self, results_dict):
        """Save to Excel"""
        print(f"\nSaving Excel report...")
        
        with pd.ExcelWriter(f'{self.output_dir}/STREAMLIT_RESULTS.xlsx', engine='openpyxl') as w:
            # All results
            all_df = pd.concat([d.assign(target=t) for t, d in results_dict.items()])
            all_df.to_excel(w, 'All_Results', index=False)
            
            # Summary
            summary = []
            for target, df in results_dict.items():
                best_idx = df['r2'].idxmax()
                summary.append({
                    'Target': target,
                    'Best_R²': df['r2'].max(),
                    'Mean_R²': df['r2'].mean(),
                    'Model': df.loc[best_idx, 'model'],
                    'Preprocessing': df.loc[best_idx, 'preprocessing'],
                })
            pd.DataFrame(summary).to_excel(w, 'Summary', index=False)
            
            # Timing
            timing_df = pd.DataFrame(self.timing_data, columns=['Step', 'Time'])
            timing_df.to_excel(w, 'Timing', index=False)
        
        print(f"✓ Excel saved: STREAMLIT_RESULTS.xlsx")
    
    def run(self):
        """Execute pipeline"""
        print("\n" + "="*70)
        print("STREAMLIT FULL PIPELINE - FAST EXECUTION")
        print("="*70)
        
        X, y = self.load_data()
        
        for i, target in enumerate(['glucose', 'Na_acetate', 'Mg_SO4']):
            df = self.process_target(X, y[:, i], target)
            self.all_results[target] = df
        
        # Visualizations
        self.generate_all_plots(self.all_results)
        
        # Excel
        self.save_excel(self.all_results)
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        for target, df in self.all_results.items():
            best = df.loc[df['r2'].idxmax()]
            print(f"\n{target}:")
            print(f"  Best R²: {best['r2']:.4f}")
            print(f"  Model: {best['model']}")
            print(f"  Preprocessing: {best['preprocessing']}")
            print(f"  Mean R²: {df['r2'].mean():.4f}")
        
        # Timing
        print("\n" + "="*70)
        print("TIMING BREAKDOWN")
        print("="*70)
        timing_df = pd.DataFrame(self.timing_data, columns=['Step', 'Time'])
        print(timing_df.to_string())
        print(f"\nTOTAL: {timing_df['Time'].sum():.3f}s")
        print("="*70 + "\n")


if __name__ == "__main__":
    pipeline = FastStreamlitPipeline()
    pipeline.run()
