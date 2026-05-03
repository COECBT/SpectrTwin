"""
Optimized Streamlit Full Pipeline with Timing & Comparison Analysis
Runs preprocessing, feature selection, and model training with detailed timing
Generates step-by-step comparison plots and timing visualization
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
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# Set Times New Roman font, no grid
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.grid'] = False
sns.set_style("whitegrid", {'grid.linestyle': ''})

class OptimizedStreamlitPipeline:
    def __init__(self):
        self.output_dir = "STREAMLIT_OPTIMIZED_RESULTS"
        Path(self.output_dir).mkdir(exist_ok=True)
        
        self.timing_data = []
        self.all_results = {}
        
    def load_and_prepare_data(self):
        """Load data and prepare for analysis"""
        start_total = time.time()
        
        print("\n" + "="*80)
        print("STEP 0: DATA LOADING & PREPARATION")
        print("="*80)
        
        # Load data
        start = time.time()
        data_path = "Data/Multi-objective verification/Data_raman.csv"
        df = pd.read_csv(data_path)
        
        target_cols = ['glucose', 'Na_acetate', 'Mg_SO4']
        X = df.drop(columns=target_cols).values
        y_data = df[target_cols].values
        
        elapsed_load = time.time() - start
        print(f"✓ Data Loading: {elapsed_load:.3f}s - {X.shape[0]} samples, {X.shape[1]} features")
        self.timing_data.append({'Step': 'Data Loading', 'Time': elapsed_load, 'Phase': 'Preparation'})
        
        return X, y_data, target_cols
    
    def run_target_pipeline(self, X, y, target_name):
        """Run pipeline for single target"""
        print(f"\n{'#'*80}")
        print(f"# ANALYZING: {target_name.upper()}")
        print(f"{'#'*80}")
        
        results_list = []
        
        # Step 1: Data Splitting
        print("\nSTEP 1A: DATA SPLITTING")
        start = time.time()
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        elapsed = time.time() - start
        print(f"  ✓ {elapsed:.3f}s - Train:Val:Test = {len(X_train)}:{len(X_val)}:{len(X_test)}")
        self.timing_data.append({'Step': f'Data Split ({target_name})', 'Time': elapsed, 'Phase': 'Preparation'})
        
        # Step 1B: Outlier Detection
        print("STEP 1B: OUTLIER DETECTION")
        start = time.time()
        Q1, Q3 = np.percentile(X_train, 25, axis=0), np.percentile(X_train, 75, axis=0)
        IQR = Q3 - Q1
        outlier_mask = ~((X_train >= Q1-1.5*IQR) & (X_train <= Q3+1.5*IQR)).all(axis=1)
        n_outliers = outlier_mask.sum()
        elapsed = time.time() - start
        print(f"  ✓ {elapsed:.3f}s - {n_outliers} outliers detected ({100*n_outliers/len(X_train):.1f}%)")
        self.timing_data.append({'Step': f'Outlier Detection ({target_name})', 'Time': elapsed, 'Phase': 'Preparation'})
        
        # Step 2: Preprocessing Methods
        print("\nSTEP 2: PREPROCESSING METHODS (8 techniques)")
        preprocess_times = {}
        preprocessed_data = {}
        
        preprocessing_methods = {
            'None': lambda X: X,
            'SNV': self._snv,
            'MinMax': self._minmax,
            'StandardScaler': self._standardscaler,
            'VectorNorm': self._vector_norm,
            'BaselineCorrection': self._baseline,
            'SG_Smoothing': self._sg_smooth,
            'SG_Derivative': self._sg_deriv,
        }
        
        for name, func in preprocessing_methods.items():
            start = time.time()
            try:
                X_train_p = func(X_train)
                X_val_p = func(X_val)
                X_test_p = func(X_test)
                elapsed = time.time() - start
                preprocessed_data[name] = (X_train_p, X_val_p, X_test_p)
                preprocess_times[name] = elapsed
                print(f"  ✓ {name:20s} {elapsed:.3f}s")
                self.timing_data.append({'Step': f'Preprocess: {name} ({target_name})', 'Time': elapsed, 'Phase': 'Preprocessing'})
            except Exception as e:
                print(f"  ✗ {name:20s} - Error")
        
        # Step 3: Feature Selection (5 levels)
        print("\nSTEP 3: FEATURE SELECTION (5 k-values)")
        feature_times = {}
        selected_features = {}
        
        for k in [50, 75, 100, 150, 200]:
            start = time.time()
            selector = SelectKBest(f_regression, k=min(k, X_train.shape[1]))
            X_train_s = selector.fit_transform(X_train, y_train)
            X_val_s = selector.transform(X_val)
            X_test_s = selector.transform(X_test)
            elapsed = time.time() - start
            selected_features[f'k_{k}'] = (X_train_s, X_val_s, X_test_s)
            feature_times[f'k_{k}'] = elapsed
            print(f"  ✓ SelectKBest(k={k:3d}) {elapsed:.3f}s")
            self.timing_data.append({'Step': f'Feature Sel (k={k}, {target_name})', 'Time': elapsed, 'Phase': 'Feature Selection'})
        
        # Step 4: Model Training (key models only - 12 selected)
        print("\nSTEP 4: MODEL TRAINING & EVALUATION (12 key models)")
        models_to_test = {
            'LinearRegression': LinearRegression(),
            'Ridge_0.01': Ridge(alpha=0.01),
            'Ridge_100': Ridge(alpha=100),
            'Lasso_0.001': Lasso(alpha=0.001, max_iter=10000),
            'ElasticNet': ElasticNet(max_iter=10000),
            'PLS_10': PLSRegression(n_components=10),
            'SVR_linear': SVR(kernel='linear'),
            'KNN_10': KNeighborsRegressor(n_neighbors=10),
            'RandomForest_100': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(random_state=42),
            'AdaBoost': AdaBoostRegressor(random_state=42),
            'BayesianRidge': BayesianRidge(),
        }
        
        model_times = {}
        total_models_time = 0
        
        for feat_name, (X_tr_f, X_v_f, X_te_f) in selected_features.items():
            print(f"\n  Feature Set: {feat_name}")
            for model_name, model in models_to_test.items():
                start = time.time()
                try:
                    model.fit(X_tr_f, y_train)
                    y_pred_test = model.predict(X_te_f)
                    test_r2 = r2_score(y_test, y_pred_test)
                    elapsed = time.time() - start
                    total_models_time += elapsed
                    
                    if test_r2 > 0.5:
                        print(f"    ✓ {model_name:20s} R²={test_r2:.3f}")
                    
                    results_list.append({
                        'target': target_name,
                        'preprocessing': 'baseline',
                        'feature_set': feat_name,
                        'model': model_name,
                        'test_r2': test_r2,
                    })
                except:
                    pass
        
        self.timing_data.append({'Step': f'Model Training ({target_name})', 'Time': total_models_time, 'Phase': 'Model Training'})
        
        return pd.DataFrame(results_list)
    
    # Preprocessing functions
    def _snv(self, X):
        mean, std = np.mean(X, axis=1, keepdims=True), np.std(X, axis=1, keepdims=True)
        return (X - mean) / (std + 1e-10)
    
    def _minmax(self, X):
        scaler = MinMaxScaler()
        return scaler.fit_transform(X)
    
    def _standardscaler(self, X):
        scaler = StandardScaler()
        return scaler.fit_transform(X)
    
    def _vector_norm(self, X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / (norms + 1e-10)
    
    def _baseline(self, X):
        X_corrected = X.copy()
        for i in range(X.shape[0]):
            baseline = np.convolve(X[i], np.ones(10)/10, mode='same')
            X_corrected[i] = X[i] - baseline
        return X_corrected
    
    def _sg_smooth(self, X, window=11, order=3):
        from scipy.signal import savgol_filter
        return np.array([savgol_filter(x, window, order) for x in X])
    
    def _sg_deriv(self, X, window=11, order=3):
        from scipy.signal import savgol_filter
        return np.array([savgol_filter(x, window, order, deriv=1) for x in X])
    
    def generate_timing_plot(self):
        """Generate comprehensive timing visualization"""
        print("\nGenerating timing visualization...")
        
        timing_df = pd.DataFrame(self.timing_data)
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 2, figure=fig)
        
        # Plot 1: All steps sorted by time
        ax1 = fig.add_subplot(gs[0, :])
        sorted_timing = timing_df.sort_values('Time', ascending=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(sorted_timing)))
        ax1.barh(range(len(sorted_timing)), sorted_timing['Time'], color=colors)
        ax1.set_yticks(range(len(sorted_timing)))
        ax1.set_yticklabels([s[:35] for s in sorted_timing['Step']], fontsize=9)
        ax1.set_xlabel('Time (seconds)', fontsize=11, family='Times New Roman', weight='bold')
        ax1.set_title('All Steps - Execution Time Breakdown', fontsize=13, family='Times New Roman', weight='bold')
        ax1.grid(False)
        
        # Add value labels
        for i, v in enumerate(sorted_timing['Time']):
            ax1.text(v + 0.001*sorted_timing['Time'].max(), i, f'{v:.3f}s', va='center', fontsize=8)
        
        # Plot 2: Time by phase
        ax2 = fig.add_subplot(gs[1, 0])
        phase_time = timing_df.groupby('Phase')['Time'].sum().sort_values(ascending=False)
        colors_pie = plt.cm.Set2(np.linspace(0, 1, len(phase_time)))
        wedges, texts, autotexts = ax2.pie(phase_time, labels=phase_time.index, autopct='%1.1f%%',
                                            colors=colors_pie, startangle=90)
        for text in texts:
            text.set_fontsize(10)
            text.set_family('Times New Roman')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(9)
            autotext.set_family('Times New Roman')
        ax2.set_title('Time Distribution by Phase', fontsize=12, family='Times New Roman', weight='bold')
        
        # Plot 3: Average time by step type
        ax3 = fig.add_subplot(gs[1, 1])
        step_base = timing_df['Step'].str.split('(').str[0].str.strip()
        step_avg = timing_df.groupby(step_base)['Time'].agg(['mean', 'std']).sort_values('mean')
        step_avg['mean'].plot(kind='barh', ax=ax3, xerr=step_avg['std'], color='coral')
        ax3.set_xlabel('Average Time (seconds)', fontsize=11, family='Times New Roman')
        ax3.set_title('Average Time by Step Type', fontsize=12, family='Times New Roman', weight='bold')
        ax3.grid(False)
        
        # Plot 4: Cumulative time
        ax4 = fig.add_subplot(gs[2, :])
        sorted_timing_chrono = timing_df.sort_values('Step')
        cumsum = sorted_timing_chrono['Time'].cumsum()
        ax4.plot(range(len(cumsum)), cumsum, marker='o', linewidth=2.5, markersize=6, color='steelblue')
        ax4.fill_between(range(len(cumsum)), cumsum, alpha=0.3, color='steelblue')
        ax4.set_xticks(range(0, len(cumsum), max(1, len(cumsum)//15)))
        ax4.set_xticklabels([s[:20] for s in sorted_timing_chrono['Step'].iloc[::max(1, len(cumsum)//15)]], rotation=45, ha='right', fontsize=8)
        ax4.set_ylabel('Cumulative Time (seconds)', fontsize=11, family='Times New Roman')
        ax4.set_title('Cumulative Execution Time Over Pipeline', fontsize=12, family='Times New Roman', weight='bold')
        ax4.grid(False)
        
        plt.tight_layout()
        output_path = f"{self.output_dir}/00_TIMING_ANALYSIS.png"
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_path}")
        
        return timing_df
    
    def generate_model_results_plot(self, target_name, results_df):
        """Generate model comparison plot"""
        print(f"Generating model results plot for {target_name}...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Best R² by model
        ax = axes[0, 0]
        model_best = results_df.loc[results_df.groupby('model')['test_r2'].idxmax()] if len(results_df) > 0 else pd.DataFrame()
        if len(model_best) > 0:
            model_best_sorted = model_best.sort_values('test_r2', ascending=False).head(10)
            model_best_sorted.set_index('model')['test_r2'].plot(kind='barh', ax=ax, color='seagreen')
            ax.set_xlabel('Best Test R²', fontsize=11, family='Times New Roman')
            ax.set_title(f'{target_name}: Best R² per Model', fontsize=12, family='Times New Roman', weight='bold')
            ax.grid(False)
        
        # Plot 2: R² by feature set
        ax = axes[0, 1]
        if len(results_df) > 0:
            feat_performance = results_df.groupby('feature_set')['test_r2'].mean().sort_values()
            feat_performance.plot(kind='barh', ax=ax, color='mediumpurple')
            ax.set_xlabel('Average Test R²', fontsize=11, family='Times New Roman')
            ax.set_title(f'{target_name}: Performance by Feature Set', fontsize=12, family='Times New Roman', weight='bold')
            ax.grid(False)
        
        # Plot 3: Distribution of R² values
        ax = axes[1, 0]
        if len(results_df) > 0:
            ax.hist(results_df['test_r2'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
            ax.axvline(results_df['test_r2'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {results_df["test_r2"].mean():.3f}')
            ax.axvline(results_df['test_r2'].max(), color='green', linestyle='--', linewidth=2, label=f'Max: {results_df["test_r2"].max():.3f}')
            ax.set_xlabel('Test R²', fontsize=11, family='Times New Roman')
            ax.set_ylabel('Frequency', fontsize=11, family='Times New Roman')
            ax.set_title(f'{target_name}: Distribution of R² Values', fontsize=12, family='Times New Roman', weight='bold')
            ax.legend()
            ax.grid(False)
        
        # Plot 4: Best configurations
        ax = axes[1, 1]
        if len(results_df) > 0:
            top_configs = results_df.nlargest(10, 'test_r2').copy()
            top_configs['config'] = top_configs['feature_set'] + '\n' + top_configs['model']
            top_configs_sorted = top_configs.sort_values('test_r2')
            ax.barh(range(len(top_configs_sorted)), top_configs_sorted['test_r2'], color='coral')
            ax.set_yticks(range(len(top_configs_sorted)))
            ax.set_yticklabels([c[:30] for c in top_configs_sorted['config']], fontsize=9)
            ax.set_xlabel('Test R²', fontsize=11, family='Times New Roman')
            ax.set_title(f'{target_name}: Top 10 Configurations', fontsize=12, family='Times New Roman', weight='bold')
            ax.grid(False)
        
        plt.tight_layout()
        output_path = f"{self.output_dir}/01_Results_{target_name}.png"
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_path}")
    
    def save_excel_results(self, all_results_df, timing_df):
        """Save all results to Excel"""
        print("\nSaving comprehensive Excel report...")
        
        excel_path = f"{self.output_dir}/STREAMLIT_FULL_ANALYSIS_RESULTS.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Sheet 1: All model results
            all_results_df.to_excel(writer, sheet_name='Model_Results', index=False)
            
            # Sheet 2: Timing data
            timing_df.to_excel(writer, sheet_name='Timing_Breakdown', index=False)
            
            # Sheet 3: Summary statistics
            summary_stats = []
            for target in all_results_df['target'].unique():
                target_df = all_results_df[all_results_df['target'] == target]
                summary_stats.append({
                    'Target': target,
                    'Best_R²': target_df['test_r2'].max(),
                    'Mean_R²': target_df['test_r2'].mean(),
                    'Std_R²': target_df['test_r2'].std(),
                    'Best_Model': target_df.loc[target_df['test_r2'].idxmax(), 'model'],
                    'Features': target_df.loc[target_df['test_r2'].idxmax(), 'feature_set'],
                    'Configs_Tested': len(target_df),
                })
            pd.DataFrame(summary_stats).to_excel(writer, sheet_name='Summary', index=False)
            
            # Sheet 4: Timing summary
            timing_summary = timing_df.groupby('Phase')['Time'].agg(['sum', 'count', 'mean']).round(3)
            timing_summary.to_excel(writer, sheet_name='Phase_Summary')
        
        print(f"✓ Saved: {excel_path}")
    
    def run_complete_analysis(self):
        """Run complete analysis on all targets"""
        print("\n" + "="*80)
        print("STREAMLIT FULL PIPELINE - COMPREHENSIVE ANALYSIS")
        print("="*80)
        
        # Load data
        X, y_data, target_cols = self.load_and_prepare_data()
        
        # Process each target
        all_results_list = []
        for idx, target_col in enumerate(target_cols):
            results = self.run_target_pipeline(X, y_data[:, idx], target_col)
            all_results_list.append(results)
            
            # Generate individual target plot
            if len(results) > 0:
                self.generate_model_results_plot(target_col, results)
                self.all_results[target_col] = results
        
        # Combine all results
        all_results_df = pd.concat(all_results_list, ignore_index=True) if all_results_list else pd.DataFrame()
        
        # Generate timing plot
        timing_df = self.generate_timing_plot()
        
        # Save to Excel
        self.save_excel_results(all_results_df, timing_df)
        
        # Print summary
        print("\n" + "="*80)
        print("PIPELINE SUMMARY")
        print("="*80)
        
        for target_col in target_cols:
            if target_col in self.all_results:
                df = self.all_results[target_col]
                best_idx = df['test_r2'].idxmax()
                best = df.loc[best_idx]
                
                print(f"\n{target_col.upper()}:")
                print(f"  Best R²: {best['test_r2']:.4f}")
                print(f"  Model: {best['model']}")
                print(f"  Features: {best['feature_set']}")
                print(f"  Mean R²: {df['test_r2'].mean():.4f} ± {df['test_r2'].std():.4f}")
        
        # Print timing summary
        print("\n" + "="*80)
        print("TIMING SUMMARY")
        print("="*80)
        phase_times = timing_df.groupby('Phase')['Time'].sum().sort_values(ascending=False)
        total_time = timing_df['Time'].sum()
        
        for phase, time_val in phase_times.items():
            print(f"  {phase:20s}: {time_val:8.3f}s ({100*time_val/total_time:5.1f}%)")
        
        print(f"  {'TOTAL':20s}: {total_time:8.3f}s")
        
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    pipeline = OptimizedStreamlitPipeline()
    pipeline.run_complete_analysis()
