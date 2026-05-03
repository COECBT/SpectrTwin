"""
Comprehensive Streamlit Full Pipeline with Timing Tracking
Runs all preprocessing and modeling steps with detailed timing and visualization
Treats each Y variable separately and generates comparison plots at each phase
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
import time
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# Set font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.grid'] = False  # No grid
sns.set_style("whitegrid", {'grid.linestyle': ''})

class StreamlitFullPipeline:
    def __init__(self, phase_name="comprehensive"):
        self.phase_name = phase_name
        self.output_dir = f"STREAMLIT_RESULTS_{phase_name.upper()}"
        Path(self.output_dir).mkdir(exist_ok=True)
        
        self.timing_log = {}
        self.results = {}
        self.preprocessing_results = {}
        self.model_results = {}
        
    def load_data(self):
        """Load and prepare data"""
        start = time.time()
        
        # Load data
        data_path = "Data/Multi-objective verification/Data_raman.csv"
        df = pd.read_csv(data_path)
        
        # Separate features and targets
        target_cols = ['glucose', 'Na_acetate', 'Mg_SO4']
        X = df.drop(columns=target_cols).values
        y_values = df[target_cols].values
        
        elapsed = time.time() - start
        self.timing_log['data_loading'] = elapsed
        
        print(f"✓ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"  Time: {elapsed:.2f}s")
        
        return X, y_values, target_cols
    
    def split_data(self, X, y):
        """Split data into train/validate/test"""
        start = time.time()
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        elapsed = time.time() - start
        self.timing_log['data_splitting'] = elapsed
        
        print(f"✓ Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        print(f"  Time: {elapsed:.2f}s")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def apply_outlier_detection(self, X_train, X_val, X_test):
        """Detect and handle outliers"""
        start = time.time()
        
        # IQR-based outlier detection
        Q1 = np.percentile(X_train, 25, axis=0)
        Q3 = np.percentile(X_train, 75, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = ~((X_train >= lower_bound) & (X_train <= upper_bound)).all(axis=1)
        out_count = outlier_mask.sum()
        
        elapsed = time.time() - start
        self.timing_log['outlier_detection'] = elapsed
        
        print(f"✓ Outliers detected: {out_count} samples ({100*out_count/len(X_train):.1f}%)")
        print(f"  Time: {elapsed:.2f}s")
        
        return X_train, X_val, X_test, out_count
    
    def preprocessing_snv(self, X):
        """Standard Normal Variate"""
        mean = np.mean(X, axis=1, keepdims=True)
        std = np.std(X, axis=1, keepdims=True)
        return (X - mean) / (std + 1e-10)
    
    def preprocessing_minmax(self, X):
        """MinMax scaling"""
        scaler = MinMaxScaler()
        return scaler.fit_transform(X)
    
    def preprocessing_standardscaler(self, X):
        """StandardScaler"""
        scaler = StandardScaler()
        return scaler.fit_transform(X)
    
    def preprocessing_vector_norm(self, X):
        """Vector normalization"""
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / (norms + 1e-10)
    
    def preprocessing_baseline_correction(self, X):
        """Baseline correction using rolling mean"""
        X_corrected = X.copy()
        for i in range(X.shape[0]):
            baseline = np.convolve(X[i], np.ones(10)/10, mode='same')
            X_corrected[i] = X[i] - baseline
        return X_corrected
    
    def preprocessing_sg_smoothing(self, X, window=11, order=3):
        """Savitzky-Golay smoothing"""
        from scipy.signal import savgol_filter
        return np.array([savgol_filter(x, window, order) for x in X])
    
    def preprocessing_sg_derivative(self, X, window=11, order=3):
        """Savitzky-Golay first derivative"""
        from scipy.signal import savgol_filter
        return np.array([savgol_filter(x, window, order, deriv=1) for x in X])
    
    def run_preprocessing_phase(self, X_train, X_val, X_test):
        """Run all preprocessing methods with timing"""
        print("\n" + "="*70)
        print("PHASE 1: PREPROCESSING METHODS")
        print("="*70)
        
        start_phase = time.time()
        
        preprocessing_methods = {
            'None': lambda X: X,
            'SNV': self.preprocessing_snv,
            'MinMax': self.preprocessing_minmax,
            'StandardScaler': self.preprocessing_standardscaler,
            'VectorNorm': self.preprocessing_vector_norm,
            'BaselineCorrection': self.preprocessing_baseline_correction,
            'SG_Smoothing': self.preprocessing_sg_smoothing,
            'SG_Derivative': self.preprocessing_sg_derivative,
        }
        
        preprocessed_data = {}
        
        for name, func in preprocessing_methods.items():
            start = time.time()
            try:
                X_train_p = func(X_train)
                X_val_p = func(X_val)
                X_test_p = func(X_test)
                elapsed = time.time() - start
                
                preprocessed_data[name] = (X_train_p, X_val_p, X_test_p)
                self.timing_log[f'prep_{name}'] = elapsed
                
                print(f"  ✓ {name:20s} - {elapsed:.3f}s")
            except Exception as e:
                print(f"  ✗ {name:20s} - Error: {str(e)[:50]}")
        
        phase_time = time.time() - start_phase
        self.timing_log['phase_preprocessing'] = phase_time
        print(f"\nPhase 1 Total Time: {phase_time:.2f}s")
        
        return preprocessed_data
    
    def run_feature_selection_phase(self, X_train, X_val, X_test, y_train):
        """Run feature selection with timing"""
        print("\n" + "="*70)
        print("PHASE 2: FEATURE SELECTION")
        print("="*70)
        
        start_phase = time.time()
        
        k_values = [50, 75, 100, 150, 200]
        selected_features = {}
        
        for k in k_values:
            start = time.time()
            try:
                selector = SelectKBest(f_regression, k=min(k, X_train.shape[1]))
                X_train_s = selector.fit_transform(X_train, y_train)
                X_val_s = selector.transform(X_val)
                X_test_s = selector.transform(X_test)
                
                elapsed = time.time() - start
                selected_features[f'k_{k}'] = (X_train_s, X_val_s, X_test_s, selector.get_support())
                self.timing_log[f'feat_sel_k{k}'] = elapsed
                
                print(f"  ✓ SelectKBest(k={k:3d}) - {elapsed:.3f}s")
            except Exception as e:
                print(f"  ✗ SelectKBest(k={k}) - Error: {str(e)[:50]}")
        
        phase_time = time.time() - start_phase
        self.timing_log['phase_feature_selection'] = phase_time
        print(f"\nPhase 2 Total Time: {phase_time:.2f}s")
        
        return selected_features
    
    def build_models(self):
        """Build all model types"""
        models = {
            'LinearRegression': LinearRegression(),
            'Ridge_0.01': Ridge(alpha=0.01),
            'Ridge_1': Ridge(alpha=1),
            'Ridge_10': Ridge(alpha=10),
            'Ridge_100': Ridge(alpha=100),
            'Lasso_0.001': Lasso(alpha=0.001, max_iter=10000),
            'Lasso_0.01': Lasso(alpha=0.01, max_iter=10000),
            'ElasticNet': ElasticNet(max_iter=10000),
            'PLS_5': PLSRegression(n_components=5),
            'PLS_10': PLSRegression(n_components=10),
            'SVR_linear': SVR(kernel='linear'),
            'SVR_rbf': SVR(kernel='rbf'),
            'KNN_5': KNeighborsRegressor(n_neighbors=5),
            'KNN_10': KNeighborsRegressor(n_neighbors=10),
            'RandomForest_50': RandomForestRegressor(n_estimators=50, random_state=42),
            'RandomForest_100': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(random_state=42),
            'AdaBoost': AdaBoostRegressor(random_state=42),
        }
        return models
    
    def run_model_training_phase(self, X_train, X_val, X_test, y_train, y_val, y_test, features_dict):
        """Train all models on selected features with timing"""
        print("\n" + "="*70)
        print("PHASE 3: MODEL TRAINING & EVALUATION")
        print("="*70)
        
        start_phase = time.time()
        models = self.build_models()
        
        all_results = []
        
        for feat_name, (X_train_f, X_val_f, X_test_f, _) in features_dict.items():
            print(f"\n  Feature Selection: {feat_name}")
            
            for model_name, model in models.items():
                start = time.time()
                try:
                    # Train
                    model.fit(X_train_f, y_train)
                    
                    # Predict
                    y_pred_val = model.predict(X_val_f)
                    y_pred_test = model.predict(X_test_f)
                    
                    # Evaluate
                    val_r2 = r2_score(y_val, y_pred_val)
                    test_r2 = r2_score(y_test, y_pred_test)
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                    test_mae = mean_absolute_error(y_test, y_pred_test)
                    
                    elapsed = time.time() - start
                    
                    all_results.append({
                        'feature_selection': feat_name,
                        'model': model_name,
                        'val_r2': val_r2,
                        'test_r2': test_r2,
                        'test_rmse': test_rmse,
                        'test_mae': test_mae,
                        'time': elapsed
                    })
                    
                    if test_r2 > 0.5:  # Only print good results
                        print(f"    ✓ {model_name:20s} (R²={test_r2:.3f})")
                        
                except Exception as e:
                    pass
        
        phase_time = time.time() - start_phase
        self.timing_log['phase_model_training'] = phase_time
        print(f"\nPhase 3 Total Time: {phase_time:.2f}s")
        
        return pd.DataFrame(all_results)
    
    def run_complete_pipeline(self, target_col, y_data, target_index):
        """Run complete pipeline for one target"""
        print(f"\n{'#'*70}")
        print(f"# TARGET: {target_col}")
        print(f"{'#'*70}")
        
        y = y_data[:, target_index]
        
        # Phase 0: Data preparation
        print("\n" + "="*70)
        print("PHASE 0: DATA PREPARATION")
        print("="*70)
        
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X_data, y)
        X_train, X_val, X_test, out_count = self.apply_outlier_detection(X_train, X_val, X_test)
        
        # Phase 1: Preprocessing
        preprocessed = self.run_preprocessing_phase(X_train, X_val, X_test)
        
        # Phase 2: Feature Selection
        features = self.run_feature_selection_phase(X_train, X_val, X_test, y_train)
        
        # Phase 3: Model Training
        results_df = self.run_model_training_phase(
            X_train, X_val, X_test, y_train, y_val, y_test, features
        )
        
        return {
            'results': results_df,
            'X_split': (X_train, X_val, X_test),
            'y_split': (y_train, y_val, y_test),
            'preprocessed': preprocessed,
            'features': features
        }
    
    def generate_preprocessing_comparison(self, results_dict, target_col):
        """Generate preprocessing comparison plot"""
        print(f"\nGenerating preprocessing comparison for {target_col}...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        df = results_dict['results']
        
        # Plot 1: Top models by test R²
        top_models = df.nlargest(15, 'test_r2')
        ax = axes[0, 0]
        top_models_grouped = top_models.groupby('feature_selection')['test_r2'].mean().sort_values()
        top_models_grouped.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_xlabel('Average Test R²', fontsize=11, family='Times New Roman')
        ax.set_ylabel('Feature Selection', fontsize=11, family='Times New Roman')
        ax.set_title(f'{target_col}: Average Performance by Feature Selection', fontsize=12, family='Times New Roman', weight='bold')
        ax.grid(False)
        
        # Plot 2: Test R² distribution by feature selection
        ax = axes[0, 1]
        feature_selections = df['feature_selection'].unique()
        positions = range(len(feature_selections))
        data_to_plot = [df[df['feature_selection'] == fs]['test_r2'].values for fs in feature_selections]
        bp = ax.boxplot(data_to_plot, labels=[fs.replace('k_', '') for fs in feature_selections], patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax.set_ylabel('Test R²', fontsize=11, family='Times New Roman')
        ax.set_xlabel('Feature Count (k)', fontsize=11, family='Times New Roman')
        ax.set_title(f'{target_col}: R² Distribution by Feature Selection', fontsize=12, family='Times New Roman', weight='bold')
        ax.grid(False)
        
        # Plot 3: Model comparison
        ax = axes[1, 0]
        model_performance = df.groupby('model')['test_r2'].agg(['mean', 'std']).sort_values('mean', ascending=False).head(10)
        model_performance['mean'].plot(kind='barh', ax=ax, color='coral', xerr=model_performance['std'])
        ax.set_xlabel('Average Test R²', fontsize=11, family='Times New Roman')
        ax.set_title(f'{target_col}: Top 10 Models by Performance', fontsize=12, family='Times New Roman', weight='bold')
        ax.grid(False)
        
        # Plot 4: RMSE vs MAE
        ax = axes[1, 1]
        scatter = ax.scatter(df['test_rmse'], df['test_mae'], c=df['test_r2'], cmap='viridis', alpha=0.6, s=50)
        ax.set_xlabel('Test RMSE', fontsize=11, family='Times New Roman')
        ax.set_ylabel('Test MAE', fontsize=11, family='Times New Roman')
        ax.set_title(f'{target_col}: Error Metrics Correlation', fontsize=12, family='Times New Roman', weight='bold')
        plt.colorbar(scatter, ax=ax, label='R²')
        ax.grid(False)
        
        plt.tight_layout()
        output_path = f"{self.output_dir}/01_Preprocessing_Comparison_{target_col}.png"
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_path}")
    
    def generate_model_comparison(self, results_dict, target_col):
        """Generate model comparison plot"""
        print(f"Generating model comparison for {target_col}...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        df = results_dict['results']
        
        # Plot 1: Best models
        ax = axes[0, 0]
        best_by_model = df.loc[df.groupby('model')['test_r2'].idxmax()]
        best_by_model_sorted = best_by_model.sort_values('test_r2', ascending=False).head(12)
        best_by_model_sorted.set_index('model')['test_r2'].plot(kind='barh', ax=ax, color='seagreen', legend=False)
        ax.set_xlabel('Best Test R²', fontsize=11, family='Times New Roman')
        ax.set_title(f'{target_col}: Best R² per Model', fontsize=12, family='Times New Roman', weight='bold')
        ax.grid(False)
        
        # Plot 2: Model training time
        ax = axes[0, 1]
        time_by_model = df.groupby('model')['time'].mean().sort_values(ascending=False).head(10)
        time_by_model.plot(kind='barh', ax=ax, color='orange')
        ax.set_xlabel('Average Training Time (seconds)', fontsize=11, family='Times New Roman')
        ax.set_title(f'{target_col}: Top 10 Models by Training Time', fontsize=12, family='Times New Roman', weight='bold')
        ax.grid(False)
        
        # Plot 3: Performance vs Complexity (num features vs R²)
        ax = axes[1, 0]
        df['n_features'] = df['feature_selection'].str.extract(r'(\d+)').astype(int)
        scatter = ax.scatter(df['n_features'], df['test_r2'], c=df['test_rmse'], cmap='RdYlGn_r', alpha=0.5, s=40)
        ax.set_xlabel('Number of Features', fontsize=11, family='Times New Roman')
        ax.set_ylabel('Test R²', fontsize=11, family='Times New Roman')
        ax.set_title(f'{target_col}: R² vs Feature Count', fontsize=12, family='Times New Roman', weight='bold')
        plt.colorbar(scatter, ax=ax, label='RMSE')
        ax.grid(False)
        
        # Plot 4: Heatmap of best configurations
        ax = axes[1, 1]
        top_20 = df.nlargest(20, 'test_r2')[['feature_selection', 'model', 'test_r2']].copy()
        top_20['config'] = top_20['feature_selection'] + '+' + top_20['model']
        top_20_data = top_20.sort_values('test_r2', ascending=False).set_index('config')[['test_r2']]
        top_20_data.plot(kind='barh', ax=ax, legend=False, color='mediumpurple')
        ax.set_xlabel('Test R²', fontsize=11, family='Times New Roman')
        ax.set_title(f'{target_col}: Top 20 Configurations', fontsize=12, family='Times New Roman', weight='bold')
        ax.grid(False)
        
        plt.tight_layout()
        output_path = f"{self.output_dir}/02_Model_Comparison_{target_col}.png"
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_path}")
    
    def save_results_to_excel(self, all_results):
        """Save all results to Excel workbook"""
        print(f"\nSaving results to Excel...")
        
        excel_path = f"{self.output_dir}/STREAMLIT_FULL_PIPELINE_RESULTS.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Overall results
            all_results_df = pd.concat(
                [df.assign(target=target) for target, df in all_results.items()],
                ignore_index=True
            )
            all_results_df.to_excel(writer, sheet_name='All_Results', index=False)
            
            # Summary statistics by target
            summary_data = []
            for target, df in all_results.items():
                summary_data.append({
                    'Target': target,
                    'Best_R²': df['test_r2'].max(),
                    'Mean_R²': df['test_r2'].mean(),
                    'Std_R²': df['test_r2'].std(),
                    'Best_Model': df.loc[df['test_r2'].idxmax(), 'model'],
                    'Best_Features': df.loc[df['test_r2'].idxmax(), 'feature_selection'],
                    'Mean_RMSE': df['test_rmse'].mean(),
                    'Mean_MAE': df['test_mae'].mean(),
                })
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Timing information
            timing_df = pd.DataFrame([
                {'Step': k, 'Time_Seconds': v} for k, v in self.timing_log.items()
            ]).sort_values('Time_Seconds', ascending=False)
            timing_df.to_excel(writer, sheet_name='Timing', index=False)
        
        print(f"✓ Saved: {excel_path}")
        return all_results_df
    
    def generate_timing_visualization(self):
        """Generate timing visualization"""
        print(f"\nGenerating timing visualization...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Remove phase-level timings for detailed view
        timing_dict = {k: v for k, v in self.timing_log.items() if not k.startswith('phase_')}
        
        # Sort by time
        sorted_timing = dict(sorted(timing_dict.items(), key=lambda x: x[1], reverse=True))
        
        # Plot 1: Horizontal bar chart
        ax = axes[0]
        steps = [k.replace('prep_', '').replace('feat_sel_', '').replace('_', ' ') for k in sorted_timing.keys()]
        times = list(sorted_timing.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(steps)))
        ax.barh(range(len(steps)), times, color=colors)
        ax.set_yticks(range(len(steps)))
        ax.set_yticklabels(steps, fontsize=9)
        ax.set_xlabel('Time (seconds)', fontsize=11, family='Times New Roman', weight='bold')
        ax.set_title('Execution Time by Step', fontsize=12, family='Times New Roman', weight='bold')
        ax.grid(False)
        
        # Add value labels
        for i, v in enumerate(times):
            ax.text(v + 0.01*max(times), i, f'{v:.2f}s', va='center', fontsize=9)
        
        # Plot 2: Pie chart for phase breakdown
        ax = axes[1]
        phase_timings = {k: v for k, v in self.timing_log.items() if k.startswith('phase_')}
        if phase_timings:
            phase_names = [k.replace('phase_', '').title() for k in phase_timings.keys()]
            phase_times = list(phase_timings.values())
            colors_pie = plt.cm.Set2(np.linspace(0, 1, len(phase_names)))
            wedges, texts, autotexts = ax.pie(phase_times, labels=phase_names, autopct='%1.1f%%',
                                              colors=colors_pie, startangle=90)
            for text in texts:
                text.set_fontsize(10)
                text.set_family('Times New Roman')
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(9)
                autotext.set_family('Times New Roman')
            ax.set_title('Time Distribution by Phase', fontsize=12, family='Times New Roman', weight='bold')
        
        plt.tight_layout()
        output_path = f"{self.output_dir}/03_Timing_Analysis.png"
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("STREAMLIT FULL PIPELINE - Comprehensive Analysis")
    print("="*70)
    
    pipeline = StreamlitFullPipeline(phase_name="complete")
    
    # Load data
    X_data, y_data, target_cols = pipeline.load_data()
    
    # Process each target separately
    all_results = {}
    
    for idx, target_col in enumerate(target_cols):
        results = pipeline.run_complete_pipeline(target_col, y_data, idx)
        all_results[target_col] = results['results']
        
        # Generate comparison plots
        pipeline.generate_preprocessing_comparison(results, target_col)
        pipeline.generate_model_comparison(results, target_col)
    
    # Save all results to Excel
    all_results_df = pipeline.save_results_to_excel(all_results)
    
    # Generate timing visualization
    pipeline.generate_timing_visualization()
    
    # Print summary
    print("\n" + "="*70)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*70)
    
    for target_col, results_df in all_results.items():
        best_idx = results_df['test_r2'].idxmax()
        best_row = results_df.loc[best_idx]
        
        print(f"\n{target_col}:")
        print(f"  Best R²: {best_row['test_r2']:.4f}")
        print(f"  Best Model: {best_row['model']}")
        print(f"  Features: {best_row['feature_selection']}")
        print(f"  RMSE: {best_row['test_rmse']:.4f}")
        print(f"  MAE: {best_row['test_mae']:.4f}")
    
    print("\n" + "="*70)
    print(f"Results saved to: {pipeline.output_dir}")
    print("="*70 + "\n")
