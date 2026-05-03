import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class MultiWayPLS:
    
    def __init__(self, X_train, X_test=None, y_train=None, y_test=None, 
                 batch_ids_train=None, batch_ids_test=None, time_col=None, feature_names=None):
        
        if isinstance(X_train, pd.DataFrame):
            if time_col and time_col in X_train.columns:
                X_train = X_train.drop(columns=[time_col])
            if batch_ids_train is None and 'Batch_ID' in X_train.columns:
                batch_ids_train = X_train['Batch_ID'].values
                X_train = X_train.drop(columns=['Batch_ID'])
            if feature_names is None:
                self.feature_names = list(X_train.columns)
            X_train = X_train.values
        
        self.batch_ids_train = batch_ids_train
        self.batch_ids_test = batch_ids_test
        
        self.X_train_3d = self._to_3d(X_train, batch_ids_train)
        self.X_test_3d = self._to_3d(X_test, batch_ids_test) if X_test is not None else None
        
        self.X_train = self._flatten_batches(self.X_train_3d)
        self.X_test = self._flatten_batches(self.X_test_3d) if self.X_test_3d is not None else None
        
        self.y_train = self._flatten_y(y_train)
        self.y_test = self._flatten_y(y_test)
        
        self.n_batches_train = self.X_train_3d.shape[0]
        self.n_timepoints = self.X_train_3d.shape[1]
        self.n_features = self.X_train_3d.shape[2]
        
        if self.X_test_3d is not None:
            self.n_batches_test = self.X_test_3d.shape[0]
        
        if feature_names is None and not hasattr(self, 'feature_names'):
            self.feature_names = [f"Var{i}" for i in range(self.n_features)]
        elif feature_names is not None:
            self.feature_names = list(feature_names)
        
        self.scaler = None
        self.pls_model = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        
        self.T_scores_train = None
        self.T_scores_test = None
        self.U_scores_train = None
        self.U_scores_test = None
        self.loadings_x = None
        self.loadings_y = None
        
        self.DModX_train = None
        self.DModX_test = None
        self.T2_train = None
        self.T2_test = None
        
        self.DModX_ucl = None
        self.DModX_lcl = None
        self.T2_ucl = None
        self.T2_lcl = None
        
        self._validate_data()
    
    def _to_3d(self, X, batch_ids):
        if X is None:
            return None
        
        X = np.asarray(X)
        
        if X.ndim == 3:
            return X
        
        if batch_ids is None:
            raise ValueError("batch_ids required to convert 2D data to 3D")
        
        unique_batches = np.unique(batch_ids)
        batch_list = []
        
        for batch in unique_batches:
            batch_data = X[batch_ids == batch]
            batch_list.append(batch_data)
        
        return np.array(batch_list)
    
    def _flatten_batches(self, X_3d):
        if X_3d is None:
            return None
        return X_3d.reshape(-1, X_3d.shape[2])
    
    def _flatten_y(self, y):
        if y is None:
            return None
        if isinstance(y, pd.Series):
            return y.values
        if isinstance(y, pd.DataFrame):
            return y.values.ravel()
        a = np.asarray(y)
        return a.ravel() if a.ndim > 1 else a
    
    def _validate_data(self):
        if self.X_test is not None and self.X_train.shape[1] != self.X_test.shape[1]:
            raise ValueError("X_train and X_test must have the same number of features")
        if self.y_train is not None and len(self.X_train) != len(self.y_train):
            raise ValueError("X_train and y_train must have the same number of samples")
        if self.y_test is not None and self.X_test is not None and len(self.X_test) != len(self.y_test):
            raise ValueError("X_test and y_test must have the same number of samples")
    
    def apply_scaling(self, scaling_method="standard"):
        if scaling_method == "standard":
            self.scaler = StandardScaler()
        elif scaling_method == "minmax":
            self.scaler = MinMaxScaler()
        elif scaling_method == "robust":
            self.scaler = RobustScaler()
        else:
            self.X_train_scaled = self.X_train.copy()
            self.X_test_scaled = self.X_test.copy() if self.X_test is not None else None
            return self.X_train_scaled, self.X_test_scaled
        
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test) if self.X_test is not None else None
        
        return self.X_train_scaled, self.X_test_scaled
    
    def fit_mpls(self, n_components=2, use_scaled=True, use_bem=True):
        X_data = self.X_train_scaled if use_scaled and self.X_train_scaled is not None else self.X_train
        
        if use_bem or self.y_train is None:
            Y_data = np.tile(np.arange(self.n_timepoints), self.n_batches_train).reshape(-1, 1)
            st.info(f"Using Batch Evolution Model (BEM) with dummy Y (time index)")
        else:
            Y_data = self.y_train.reshape(-1, 1) if self.y_train.ndim == 1 else self.y_train
        
        self.pls_model = PLSRegression(n_components=n_components)
        self.pls_model.fit(X_data, Y_data)
        
        self.T_scores_train = self.pls_model.x_scores_
        self.U_scores_train = self.pls_model.y_scores_
        self.loadings_x = np.ascontiguousarray(self.pls_model.x_loadings_)
        self.loadings_y = np.ascontiguousarray(self.pls_model.y_loadings_)
        
        self._compute_monitoring_stats(X_data, self.T_scores_train, is_train=True)
        
        if self.X_test is not None:
            X_test_data = self.X_test_scaled if use_scaled and self.X_test_scaled is not None else self.X_test
            self.T_scores_test = self.pls_model.transform(X_test_data)
            self._compute_monitoring_stats(X_test_data, self.T_scores_test, is_train=False)
        
        st.success(f"MPLS model fitted with {n_components} components")
        return self
    
    def _compute_monitoring_stats(self, X_data, T_scores, is_train=True):
        X_hat = T_scores @ self.loadings_x.T
        residuals = X_data - X_hat
        DModX_flat = np.sqrt(np.sum(residuals**2, axis=1))
        
        T_std = np.std(self.T_scores_train, axis=0)
        T_std[T_std == 0] = 1
        T2_flat = np.sum((T_scores / T_std)**2, axis=1)
        
        if is_train:
            self.DModX_train = DModX_flat.reshape(self.n_batches_train, self.n_timepoints)
            self.T2_train = T2_flat.reshape(self.n_batches_train, self.n_timepoints)
            self._compute_control_limits()
        else:
            total_samples = len(DModX_flat)
            
            if total_samples < self.n_timepoints:
                st.warning(f"Test data ({total_samples} samples) is less than timepoints ({self.n_timepoints}). Treating as 1 incomplete batch.")
                self.DModX_test = DModX_flat.reshape(1, -1)  # (1, total_samples)
                self.T2_test = T2_flat.reshape(1, -1)
                self.n_batches_test = 1
                self.n_timepoints_test = total_samples  # Store actual test timepoints
                return
            
            if total_samples % self.n_timepoints != 0:
                st.warning(f"Test data ({total_samples} samples) not evenly divisible by timepoints ({self.n_timepoints})")
            
            n_batches_test = total_samples // self.n_timepoints
            if n_batches_test == 0:
                n_batches_test = 1
                truncated_samples = total_samples
                self.n_timepoints_test = total_samples
            else:
                truncated_samples = n_batches_test * self.n_timepoints
                DModX_flat = DModX_flat[:truncated_samples]
                T2_flat = T2_flat[:truncated_samples]
                self.n_timepoints_test = self.n_timepoints
            
            self.DModX_test = DModX_flat.reshape(n_batches_test, -1)
            self.T2_test = T2_flat.reshape(n_batches_test, -1)
            self.n_batches_test = n_batches_test

    def _compute_control_limits(self):
        DModX_mean = np.mean(self.DModX_train, axis=0)
        DModX_std = np.std(self.DModX_train, axis=0)
        self.DModX_ucl = DModX_mean + 3 * DModX_std
        self.DModX_lcl = DModX_mean - 3 * DModX_std
        
        T2_mean = np.mean(self.T2_train, axis=0)
        T2_std = np.std(self.T2_train, axis=0)
        self.T2_ucl = T2_mean + 3 * T2_std
        self.T2_lcl = T2_mean - 3 * T2_std
    
    def plot_spc_chart(self, metric='DModX', use_train=True, batch_labels=None, 
                       title=None, highlight_ooc=True):
        if self.pls_model is None:
            st.error("No MPLS model fitted. Run fit_mpls() first.")
            return
        
        if metric == 'DModX':
            data = self.DModX_train if use_train else self.DModX_test
            ucl = self.DModX_ucl
            lcl = self.DModX_lcl
            ylabel = "DModX (Normalized Distance)"
        elif metric == 'T2':
            data = self.T2_train if use_train else self.T2_test
            ucl = self.T2_ucl
            lcl = self.T2_lcl
            ylabel = "Hotelling's T²"
        elif metric == 'score':
            scores_3d = self.T_scores_train.reshape(self.n_batches_train, self.n_timepoints, -1) if use_train else \
                        self.T_scores_test.reshape(-1, self.n_timepoints, self.pls_model.n_components)
            data = scores_3d[:, :, 0]
            score_mean = np.mean(data, axis=0)
            score_std = np.std(data, axis=0)
            ucl = score_mean + 3 * score_std
            lcl = score_mean - 3 * score_std
            ylabel = "Score t[1]"
        else:
            st.error(f"Unknown metric: {metric}")
            return
        
        mean_line = np.mean(data, axis=0)
        
        if use_train:
            time = np.arange(self.n_timepoints)
        else:
            actual_timepoints = data.shape[1]
            time = np.arange(actual_timepoints)
            if len(ucl) > actual_timepoints:
                ucl = ucl[:actual_timepoints]
                lcl = lcl[:actual_timepoints]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        n_batches = data.shape[0]
        palette = plt.cm.tab10(np.linspace(0, 1, n_batches))
        
        for i in range(n_batches):
            label = batch_labels[i] if batch_labels else f"Batch {i+1}"
            ax.plot(time, data[i], label=label, color=palette[i], linewidth=2)
            
            if highlight_ooc:
                ooc_idx = np.where((data[i] > ucl) | (data[i] < lcl))[0]
                if len(ooc_idx) > 0:
                    ax.scatter(time[ooc_idx], data[i][ooc_idx], 
                             color='red', s=60, edgecolors='black', zorder=5)
        
        ax.plot(time, mean_line, 'g--', label="Average", linewidth=2)
        ax.plot(time, ucl, 'r--', label="+3σ (UCL)", linewidth=2)
        ax.plot(time, lcl, 'r--', label="-3σ (LCL)", linewidth=2)
        
        title = title or f"{metric} - SPC Chart ({'Training' if use_train else 'Test'} Batches)"
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel("Time", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    def plot_loadings(self, components=[1, 2], title="PLS Loadings"):
        if self.loadings_x is None:
            st.error("No PLS loadings available. Run fit_mpls() first.")
            return
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        x_axis = np.arange(self.n_features)
        
        for comp in components:
            if comp <= self.pls_model.n_components:
                ax.plot(x_axis, self.loadings_x[:, comp-1], 
                       label=f"Component {comp}", linewidth=2, marker='o', markersize=4)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel("Feature Index", fontsize=14)
        ax.set_ylabel("Loading Value", fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    def plot_scores_scatter(self, components=(1, 2), use_train=True, batch_labels=None):
        if self.T_scores_train is None:
            st.error("No PLS scores available. Run fit_mpls() first.")
            return
        
        comp1, comp2 = components[0] - 1, components[1] - 1
        
        scores = self.T_scores_train if use_train else self.T_scores_test
        n_batches = self.n_batches_train if use_train else self.n_batches_test
        
        scores_3d = scores.reshape(n_batches, self.n_timepoints, -1)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        palette = plt.cm.tab10(np.linspace(0, 1, n_batches))
        
        for i in range(n_batches):
            label = batch_labels[i] if batch_labels else f"Batch {i+1}"
            ax.scatter(scores_3d[i, :, comp1], scores_3d[i, :, comp2],
                      label=label, color=palette[i], s=50, alpha=0.7)
        
        ax.set_xlabel(f"Component {comp1+1}", fontsize=14)
        ax.set_ylabel(f"Component {comp2+1}", fontsize=14)
        ax.set_title(f"PLS Scores: Component {comp1+1} vs {comp2+1}", fontsize=16, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    def plot_explained_variance(self):
        if self.pls_model is None:
            st.error("No MPLS model fitted. Run fit_mpls() first.")
            return
        
        X_data = self.X_train_scaled if self.X_train_scaled is not None else self.X_train
        
        total_var = np.var(X_data, axis=0).sum()
        var_explained = []
        
        for i in range(self.pls_model.n_components):
            T_i = self.T_scores_train[:, i:i+1]
            P_i = self.loadings_x[:, i:i+1].T
            X_reconstructed = T_i @ P_i
            var_comp = np.var(X_reconstructed, axis=0).sum()
            var_explained.append(var_comp / total_var)
        
        cumsum_var = np.cumsum(var_explained)
        
        fig, ax = plt.subplots(figsize=(9, 5))
        idx = np.arange(1, len(var_explained) + 1)
        
        ax.bar(idx, var_explained, alpha=0.6, label='Individual', color='skyblue')
        ax.plot(idx, cumsum_var, 'o-', color='darkblue', label='Cumulative', linewidth=2)
        
        ax.set_xlabel('PLS Component', fontsize=14)
        ax.set_ylabel('Explained Variance Ratio', fontsize=14)
        ax.set_title('PLS Explained Variance', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    def get_scores(self, use_train=True):
        scores = self.T_scores_train if use_train else self.T_scores_test
        if scores is None:
            return None
        
        n_comp = self.pls_model.n_components
        cols = [f"t{i+1}" for i in range(n_comp)]
        return pd.DataFrame(scores, columns=cols)
    
    def get_loadings(self):
        if self.loadings_x is None:
            return None
        
        loadings_raw = self.pls_model.x_loadings_
        
        n_comp = self.pls_model.n_components
        n_feat = self.n_features
        cols = [f"Comp{i+1}" for i in range(n_comp)]
        
        if loadings_raw.shape != (n_feat, n_comp):
            loadings_data = loadings_raw.T
        else:
            loadings_data = loadings_raw
        
        loadings_clean = np.array(loadings_data, copy=True, order='C')
        
        assert loadings_clean.shape[0] == len(self.feature_names), f"Rows {loadings_clean.shape[0]} != features {len(self.feature_names)}"
        assert loadings_clean.shape[1] == len(cols), f"Cols {loadings_clean.shape[1]} != components {len(cols)}"
        
        return pd.DataFrame(loadings_clean, index=self.feature_names, columns=cols)

    
    def get_monitoring_stats(self, use_train=True):
        if use_train:
            DModX = self.DModX_train
            T2 = self.T2_train
            n_batches = self.n_batches_train
            n_timepoints = self.n_timepoints
        else:
            DModX = self.DModX_test
            T2 = self.T2_test
            n_batches = self.n_batches_test if self.X_test is not None else 0
            n_timepoints = self.n_timepoints_test if hasattr(self, 'n_timepoints_test') else self.n_timepoints
        
        if DModX is None or T2 is None:
            return None
        
        data = []
        for batch_idx in range(n_batches):
            for time_idx in range(DModX.shape[1]):  # Use actual shape
                data.append({
                    'Batch': batch_idx + 1,
                    'Time': time_idx,
                    'DModX': DModX[batch_idx, time_idx],
                    'T²': T2[batch_idx, time_idx]
                })
        
        return pd.DataFrame(data)

    
    def detect_outliers(self, use_train=True):
        stats_df = self.get_monitoring_stats(use_train=use_train)
        if stats_df is None:
            return None
        
        stats_df['DModX_UCL'] = np.tile(self.DModX_ucl, stats_df['Batch'].nunique())
        stats_df['DModX_LCL'] = np.tile(self.DModX_lcl, stats_df['Batch'].nunique())
        stats_df['T²_UCL'] = np.tile(self.T2_ucl, stats_df['Batch'].nunique())  
        stats_df['T²_LCL'] = np.tile(self.T2_lcl, stats_df['Batch'].nunique())  
        
        stats_df['DModX_OOC'] = (stats_df['DModX'] > stats_df['DModX_UCL']) | (stats_df['DModX'] < stats_df['DModX_LCL'])
        stats_df['T²_OOC'] = (stats_df['T²'] > stats_df['T²_UCL']) | (stats_df['T²'] < stats_df['T²_LCL'])  
        
        outliers = stats_df[stats_df['DModX_OOC'] | stats_df['T²_OOC']]  
        return outliers

    
    def reset(self):
        self.scaler = None
        self.pls_model = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.T_scores_train = None
        self.T_scores_test = None
        self.U_scores_train = None
        self.U_scores_test = None
        self.loadings_x = None
        self.loadings_y = None
        self.DModX_train = None
        self.DModX_test = None
        self.T2_train = None
        self.T2_test = None
        self.DModX_ucl = None
        self.DModX_lcl = None
        self.T2_ucl = None
        self.T2_lcl = None
        
        st.success("All transformations reset")