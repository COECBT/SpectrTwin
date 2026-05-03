import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel, VarianceThreshold, mutual_info_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_consistent_length


def _center_scale_xy(X, Y, scale=True):
    X = X.copy()
    Y = Y.copy()
    x_mean = X.mean(axis=0)
    X -= x_mean
    y_mean = Y.mean(axis=0)
    Y -= y_mean
    if scale:
        x_std = X.std(axis=0, ddof=1)
        x_std[x_std == 0.0] = 1.0
        X /= x_std
        y_std = Y.std(axis=0, ddof=1)
        y_std[y_std == 0.0] = 1.0
        Y /= y_std
    else:
        x_std = np.ones(X.shape[1])
        y_std = np.ones(Y.shape[1])
    return X, Y, x_mean, y_mean, x_std, y_std


class OPLS(BaseEstimator, TransformerMixin):

    def __init__(self, X_train, X_test, y_train, y_test, n_components=5, scale=True):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.n_components = n_components
        self.scale = scale

        self.W_ortho_ = None
        self.P_ortho_ = None
        self.T_ortho_ = None

        self.x_mean_ = None
        self.y_mean_ = None
        self.x_std_ = None
        self.y_std_ = None

    def fit(self, X=None, Y=None):
        if X is None:
            X = self.X_train
        if Y is None:
            Y = self.y_train

        check_consistent_length(X, Y)
        X = check_array(X, dtype=np.float64, copy=True, ensure_min_samples=2)
        Y = check_array(Y, dtype=np.float64, copy=True, ensure_2d=False)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        X, Y, self.x_mean_, self.y_mean_, self.x_std_, self.y_std_ = _center_scale_xy(X, Y, self.scale)

        Z = X.copy()
        w = np.dot(X.T, Y)  
        w /= np.linalg.norm(w)  
        W_ortho = []
        T_ortho = []
        P_ortho = []

        for i in range(self.n_components):
            t = np.dot(Z, w)
            p = np.dot(Z.T, t) / np.dot(t.T, t).item()
            w_ortho = p - np.dot(w.T, p).item() / np.dot(w.T, w).item() * w
            w_ortho = w_ortho / np.linalg.norm(w_ortho)
            t_ortho = np.dot(Z, w_ortho)
            p_ortho = np.dot(Z.T, t_ortho) / np.dot(t_ortho.T, t_ortho).item()
            Z -= np.dot(t_ortho, p_ortho.T)
            W_ortho.append(w_ortho)
            T_ortho.append(t_ortho)
            P_ortho.append(p_ortho)

        self.W_ortho_ = np.hstack(W_ortho)
        self.T_ortho_ = np.hstack(T_ortho)
        self.P_ortho_ = np.hstack(P_ortho)

        return self

    def transform(self, X=None):
        if X is None:
            X_train_transformed = self._transform_single(self.X_train)
            X_test_transformed = self._transform_single(self.X_test)
            return X_train_transformed, X_test_transformed
        
        return self._transform_single(X)

    def _transform_single(self, X):
        Z = check_array(X, copy=True)

        Z -= self.x_mean_
        if self.scale:
            Z /= self.x_std_

        for i in range(self.n_components):
            t = np.dot(Z, self.W_ortho_[:, i]).reshape(-1, 1)
            Z -= np.dot(t, self.P_ortho_[:, i].T.reshape(1, -1))

        return Z

    def fit_transform(self, X=None, y=None, **fit_params):
        if X is None and y is None:
            return self.fit().transform()
        return self.fit(X, y).transform(X)

    def score(self, X):
        X = check_array(X)
        Z = self._transform_single(X)
        return np.sum(np.square(Z)) / np.sum(np.square(X - self.x_mean_))

    def plot_components(self, sample_idx=0):
        if not hasattr(self, 'T_ortho_') or self.T_ortho_ is None:
            print("Model not fitted yet. Call fit() first.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0, 0].plot(self.T_ortho_[sample_idx, :])
        axes[0, 0].set_title('Orthogonal Components')
        axes[0, 0].set_xlabel('Component')
        axes[0, 0].set_ylabel('Score')

        original = self.X_train[sample_idx, :]
        transformed_train, _ = self.transform()
        transformed = transformed_train[sample_idx, :]

        axes[0, 1].plot(original, 'b-', alpha=0.7, label='Original')
        axes[0, 1].plot(transformed, 'r-', alpha=0.8, label='OPLS Filtered')
        axes[0, 1].set_title('Original vs OPLS Filtered')
        axes[0, 1].set_xlabel('Feature Index')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].legend()

        axes[1, 0].bar(range(self.n_components), np.var(self.T_ortho_, axis=0))
        axes[1, 0].set_title('Component Variance')
        axes[1, 0].set_xlabel('Component')
        axes[1, 0].set_ylabel('Variance')

        loadings = self.W_ortho_[:, 0] if self.W_ortho_.shape[1] > 0 else []
        axes[1, 1].plot(loadings)
        axes[1, 1].set_title('First Component Loadings')
        axes[1, 1].set_xlabel('Feature Index')
        axes[1, 1].set_ylabel('Loading')

        plt.tight_layout()
        plt.show()

    def plot_comparison(self, sample_idx=0):
        if not hasattr(self, 'x_mean_') or self.x_mean_ is None:
            print("Model not fitted yet. Call fit() first.")
            return

        transformed_train, transformed_test = self.transform()
        
        plt.figure(figsize=(15, 6))
        
        original_train = self.X_train[sample_idx, :]
        filtered_train = transformed_train[sample_idx, :]
        
        plt.plot(original_train, 'b-', alpha=0.7, label='Original', linewidth=2)
        plt.plot(filtered_train, 'r-', alpha=0.8, label='OPLS Filtered', linewidth=2)
        
        plt.title(f"OPLS Filtering Comparison - Sample {sample_idx}", size=16)
        plt.xlabel("Feature Index", size=14)
        plt.ylabel("Intensity", size=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()