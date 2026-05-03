import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel, VarianceThreshold, mutual_info_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class DimensionalityReduction:
    def __init__(self, X_train, X_test, y_train=None, y_test=None):
 
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        # Transformation components
        self.scaler = None
        self.reducer = None
        self.feature_names = None
        self.explained_variance_ratio_ = None
        
        # Store scaled versions
        self.X_train_scaled = None
        self.X_test_scaled = None
        
        # Store transformed versions
        self.X_train_reduced = None
        self.X_test_reduced = None
        
        # Validate input data
        self._validate_data()
        
    def _validate_data(self):
        """Validate the input data"""
        if self.X_train.shape[1] != self.X_test.shape[1]:
            raise ValueError("X_train and X_test must have the same number of features")
        
        if self.y_train is not None:
            if len(self.X_train) != len(self.y_train):
                raise ValueError("X_train and y_train must have the same number of samples")
                
        if self.y_test is not None:
            if len(self.X_test) != len(self.y_test):
                raise ValueError("X_test and y_test must have the same number of samples")
        
    def apply_scaling(self, scaling_method="standard"):
        """Apply different scaling methods to stored data"""
        if scaling_method == "standard":
            self.scaler = StandardScaler()
        elif scaling_method == "minmax":
            self.scaler = MinMaxScaler()
        elif scaling_method == "robust":
            self.scaler = RobustScaler()
        else:
            self.X_train_scaled = self.X_train.copy()
            self.X_test_scaled = self.X_test.copy()
            return self.X_train_scaled, self.X_test_scaled
            
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        return self.X_train_scaled, self.X_test_scaled
    
    def pca_analysis(self, method="variance", n_components=None, variance_threshold=0.95, use_scaled=True):
        """Enhanced PCA with multiple selection criteria"""
        
        # Choose data source
        X_train_data = self.X_train_scaled if use_scaled and self.X_train_scaled is not None else self.X_train
        X_test_data = self.X_test_scaled if use_scaled and self.X_test_scaled is not None else self.X_test
        
        if method == "variance":
            pca = PCA()
            pca.fit(X_train_data)
            
            cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
            
            st.write(f"Selected {n_components} components for {variance_threshold*100}% variance")
            
        elif method == "elbow":
            pca = PCA()
            pca.fit(X_train_data)
            
            # Calculate differences to find elbow
            variance_ratios = pca.explained_variance_ratio_
            diffs = np.diff(variance_ratios)
            n_components = np.argmin(diffs) + 2  # +2 because of diff and 0-indexing
            
            st.write(f"Elbow method selected {n_components} components")
            
        elif method == "fixed":
            if n_components is None:
                n_components = min(10, X_train_data.shape[1])
        
        # Apply PCA
        self.reducer = PCA(n_components=n_components)
        self.X_train_reduced = self.reducer.fit_transform(X_train_data)
        self.X_test_reduced = self.reducer.transform(X_test_data)
        
        self.explained_variance_ratio_ = self.reducer.explained_variance_ratio_
        
        return self.X_train_reduced, self.X_test_reduced, n_components

    def feature_selection(self, method="selectkbest", use_scaled=True, **kwargs):
        """Feature selection instead of dimensionality reduction"""
        
        if self.y_train is None and method in ['selectkbest', 'rfe', 'model_based']:
            raise ValueError(f"y_train is required for {method} method")
        
        # Choose data source
        X_train_data = self.X_train_scaled if use_scaled and self.X_train_scaled is not None else self.X_train
        X_test_data = self.X_test_scaled if use_scaled and self.X_test_scaled is not None else self.X_test
        
        if method == "selectkbest":
            k = kwargs.get('k', min(500, X_train_data.shape[1]))
            score_func = kwargs.get('score_func', mutual_info_regression)
            
            self.reducer = SelectKBest(score_func=score_func, k=k)
            self.X_train_reduced = self.reducer.fit_transform(X_train_data, self.y_train)
            self.X_test_reduced = self.reducer.transform(X_test_data)
            
        elif method == "rfe":
            estimator = kwargs.get('estimator', RandomForestRegressor(n_estimators=50, random_state=42))
            n_features = kwargs.get('n_features', min(500, X_train_data.shape[1]))
            
            self.reducer = RFE(estimator=estimator, n_features_to_select=n_features)
            self.X_train_reduced = self.reducer.fit_transform(X_train_data, self.y_train)
            self.X_test_reduced = self.reducer.transform(X_test_data)
            
        elif method == "model_based":
            estimator = kwargs.get('estimator', RandomForestRegressor(n_estimators=50, random_state=42))
            threshold = kwargs.get('threshold', 'mean')
            
            self.reducer = SelectFromModel(estimator=estimator, threshold=threshold)
            self.X_train_reduced = self.reducer.fit_transform(X_train_data, self.y_train)
            self.X_test_reduced = self.reducer.transform(X_test_data)
            
        elif method == "variance_threshold":
            threshold = kwargs.get('threshold', 0.01)
            
            self.reducer = VarianceThreshold(threshold=threshold)
            self.X_train_reduced = self.reducer.fit_transform(X_train_data)
            self.X_test_reduced = self.reducer.transform(X_test_data)
            
        return self.X_train_reduced, self.X_test_reduced
    
    def plot_explained_variance(self):
        """Plot explained variance for PCA"""
        if self.explained_variance_ratio_ is None:
            st.error("No PCA results to plot. Run PCA analysis first.")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Individual explained variance
        ax1.bar(range(1, len(self.explained_variance_ratio_) + 1), 
                self.explained_variance_ratio_)
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Individual Explained Variance')
        
        # Cumulative explained variance
        cumsum_variance = np.cumsum(self.explained_variance_ratio_)
        ax2.plot(range(1, len(cumsum_variance) + 1), cumsum_variance, 'bo-')
        ax2.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('Cumulative Explained Variance')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        st.pyplot(fig)
        
    def plot_3d_visualization(self, use_train=True, title="3D Visualization"):
        """Create interactive 3D plot"""
        
        if self.X_train_reduced is None:
            st.error("No reduced data available. Run a dimensionality reduction method first.")
            return
        
        X_data = self.X_train_reduced if use_train else self.X_test_reduced
        y_data = self.y_train if use_train else self.y_test
        
        if X_data.shape[1] < 3:
            st.warning("Need at least 3 components for 3D visualization")
            return
            
        df = pd.DataFrame(X_data[:, :3], columns=['Component 1', 'Component 2', 'Component 3'])
        
        if y_data is not None:
            df['Target'] = y_data
            fig = px.scatter_3d(df, x='Component 1', y='Component 2', z='Component 3', 
                              color='Target', title=f"{title} ({'Training' if use_train else 'Test'} Data)")
        else:
            fig = px.scatter_3d(df, x='Component 1', y='Component 2', z='Component 3', 
                              title=f"{title} ({'Training' if use_train else 'Test'} Data)")
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_2d_visualization(self, use_train=True, title="2D Visualization"):
        """Create interactive 2D plot"""
        
        if self.X_train_reduced is None:
            st.error("No reduced data available. Run a dimensionality reduction method first.")
            return
        
        X_data = self.X_train_reduced if use_train else self.X_test_reduced
        y_data = self.y_train if use_train else self.y_test
        
        if X_data.shape[1] < 2:
            st.warning("Need at least 2 components for 2D visualization")
            return
            
        df = pd.DataFrame(X_data[:, :2], columns=['Component 1', 'Component 2'])
        
        if y_data is not None:
            df['Target'] = y_data
            fig = px.scatter(df, x='Component 1', y='Component 2', 
                           color='Target', title=f"{title} ({'Training' if use_train else 'Test'} Data)")
        else:
            fig = px.scatter(df, x='Component 1', y='Component 2', 
                           title=f"{title} ({'Training' if use_train else 'Test'} Data)")
        
        st.plotly_chart(fig, use_container_width=True)
    
    def compare_methods(self, methods=['pca', 'feature_selection'], use_scaled=True, **kwargs):
        """Compare PCA and feature selection methods"""
        results = {}
        
        # Ensure we have scaled data if requested
        if use_scaled and self.X_train_scaled is None:
            self.apply_scaling()
        
        for method in methods:
            try:
                if method == 'pca':
                    X_train_red, X_test_red, n_comp = self.pca_analysis(
                        method="variance", variance_threshold=0.95, use_scaled=use_scaled
                    )
                elif method == 'feature_selection' and self.y_train is not None:
                    fs_method = kwargs.get('fs_method', 'selectkbest')
                    X_train_red, X_test_red = self.feature_selection(
                        method=fs_method, use_scaled=use_scaled, **kwargs
                    )
                    n_comp = X_train_red.shape[1]
                else:
                    st.warning(f"Skipping {method}: requires y_train or not implemented")
                    continue
                    
                results[method] = {
                    'X_train': X_train_red,
                    'X_test': X_test_red,
                    'shape': X_train_red.shape,
                    'n_components': n_comp
                }
                
                st.write(f"{method.upper()}: {X_train_red.shape[0]} samples → {X_train_red.shape[1]} features")
                
            except Exception as e:
                st.error(f"Error with {method}: {str(e)}")
                
        return results
    
    def get_feature_importance(self):
        """Get feature importance for feature selection methods"""
        if self.reducer is None:
            st.error("No reducer fitted yet")
            return None
            
        if hasattr(self.reducer, 'scores_'):
            # For SelectKBest
            return self.reducer.scores_
        elif hasattr(self.reducer, 'ranking_'):
            # For RFE
            return self.reducer.ranking_
        elif hasattr(self.reducer, 'estimator_') and hasattr(self.reducer.estimator_, 'feature_importances_'):
            # For SelectFromModel with tree-based models
            return self.reducer.estimator_.feature_importances_
        else:
            st.info("Feature importance not available for this method")
            return None
    
    def inverse_transform(self, use_train=True):
        """Inverse transform for methods that support it"""
        if self.reducer is None:
            st.error("No reducer fitted yet")
            return None
            
        X_data = self.X_train_reduced if use_train else self.X_test_reduced
        
        if hasattr(self.reducer, 'inverse_transform'):
            return self.reducer.inverse_transform(X_data)
        else:
            st.warning("This method doesn't support inverse transform")
            return None
    
    def get_transformed_data(self, use_train=True):
        """Get the transformed data"""
        if self.X_train_reduced is None:
            st.error("No transformed data available. Run a dimensionality reduction method first.")
            return None
        
        if use_train:
            return self.X_train_reduced
        else:
            return self.X_test_reduced
    
    def reset(self):
        """Reset all transformations"""
        self.scaler = None
        self.reducer = None
        self.explained_variance_ratio_ = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.X_train_reduced = None
        self.X_test_reduced = None
        st.success("All transformations reset")