import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pywt
import streamlit as st
import pickle
import json
import datetime
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

import math
from sklearn.metrics import (
    accuracy_score, f1_score, r2_score,
    mean_absolute_error, mean_squared_error,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)

from sklearn.model_selection import (
    train_test_split, cross_val_score,
    KFold, StratifiedKFold
)

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import (
    LinearRegression, LogisticRegression, Lasso,
    Ridge, ElasticNet, SGDClassifier, SGDRegressor,
    RidgeClassifier
)

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    BaggingClassifier, BaggingRegressor
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct, WhiteKernel
from sklearn.svm import SVC, SVR

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier

import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor

import optuna
import optuna.visualization as vis
from bayes_opt import BayesianOptimization
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge


class ReadingData:
    def __init__(self):
        pass

    def csv_file(self, file_path):
        data = pd.read_csv(file_path)
        return data

    def xlsx_file(self, file_path):
        data = pd.read_excel(file_path)
        return data

    def text_file(self, file_path):
        with open(file_path, 'r') as file:
            data = file.read()
        return data
    
    def read_data(self, file_path):  
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".csv":
            return self.csv_file(file_path)
        elif ext == ".xlsx":
            return self.xlsx_file(file_path)
        elif ext == ".txt":
            return self.text_file(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def train_test_split_data(self, data, test_size=0.2, random_state=42):
        if isinstance(data, pd.DataFrame):
            X = data.drop('target_column', axis=1) 
            y = data['target_column'] 
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            return X_train, X_test, y_train, y_test
        else:
            raise ValueError("Data is not in the correct format for train-test split.")


class Models:
    def __init__(self):
        self.model = None
        self.predictions = None

    def get_param(self, prompt, default, cast_type):
        try:
            value = input(f"{prompt} (default={default}): ")
            return cast_type(value) if value.strip() else default
        except:
            return default

    def fit_predict_evaluate(self, model, X_train, X_test, y_train, y_test, is_classification=True):
        model.fit(X_train, y_train)
        self.predictions = model.predict(X_test)
        self.model = model

        if is_classification:
            acc = accuracy_score(y_test, self.predictions)
            f1 = f1_score(y_test, self.predictions, average="weighted")

            single_traget = (len(y_train.shape ) == 1) or (y_train.shape[1] == 1)
            
            if single_traget:
                st.write("### Classification Report:")
                st.text(classification_report(y_test, self.predictions))

                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                # Confusion Matrix
                cm = confusion_matrix(y_test, self.predictions)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(ax=axes[0], cmap="Blues")
                axes[0].set_title("Confusion Matrix")

                # ROC Curve (if binary)
                if len(np.unique(y_test)) == 2:
                    try:
                        y_prob = self.model.predict_proba(X_test)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, y_prob)
                        auc_score = roc_auc_score(y_test, y_prob)

                        axes[1].plot(fpr, tpr, label=f"AUC = {auc_score:.2f}", color="blue")
                        axes[1].plot([0, 1], [0, 1], 'r--', label="Random Guess")
                        axes[1].set_xlabel("False Positive Rate")
                        axes[1].set_ylabel("True Positive Rate")
                        axes[1].set_title("ROC AUC Curve")
                        axes[1].legend()
                        axes[1].grid(True)
                    except Exception as e:
                        axes[1].text(0.5, 0.5, "ROC AUC plot skipped\nmodel does not support `predict_proba`.",
                                    ha='center', va='center', fontsize=12)
                        axes[1].axis('off')
                else:
                    axes[1].text(0.5, 0.5, "ROC AUC only for binary classification", ha='center', va='center', fontsize=12)
                    axes[1].axis('off')

                plt.tight_layout()
                st.pyplot(fig)
                plt.clf()
                st.write(f"acc: **{acc:.4f}**, f1: **{f1:.4f}**")

                return model, self.predictions, acc, f1
            
            else :
                st.write(f"acc: **{acc:.4f}**, f1: **{f1:.4f}**")
                return model, self.predictions, acc, f1

        else:
            r2 = r2_score(y_test, self.predictions)
            mae = mean_absolute_error(y_test, self.predictions)
            mse = mean_squared_error(y_test, self.predictions)
            rmse = np.sqrt(mse)

            single_traget = (len(y_train.shape )==1) or (y_train.shape[1] == 1)

            if single_traget:


                st.write(f"### Regression Metrics:")

                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                # Actual vs Predicted scatter plot
                sns.scatterplot(x=y_test, y=self.predictions, ax=axes[0])
                axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                axes[0].set_xlabel("Actual Values")
                axes[0].set_ylabel("Predicted Values")
                axes[0].set_title("Actual vs Predicted")
                axes[0].grid(True)

                # Residuals histogram
                residuals = y_test - self.predictions
                sns.histplot(residuals, bins=30, kde=True, ax=axes[1])
                axes[1].set_title("Residuals Distribution")
                axes[1].set_xlabel("Residuals")
                axes[1].set_ylabel("Frequency")
                axes[1].grid(True)

                plt.tight_layout()
                st.pyplot(fig)
                plt.clf()

                st.write(f"R²: **{r2:.4f}**, MAE: **{mae:.4f}**, MSE: **{mse:.4f}**, RMSE: **{rmse:.4f}**")
                return model, self.predictions, r2, mae, mse, rmse
            
            else :
                st.write(f"R²: **{r2:.4f}**, MAE: **{mae:.4f}**, MSE: **{mse:.4f}**, RMSE: **{rmse:.4f}**")
                return model, self.predictions, r2, mae, mse, rmse

    def k_fold_cross_validation(self, model, X, y, n_splits=5, is_classification=True):
        if is_classification:
            from sklearn.model_selection import StratifiedKFold
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        scores = []

        for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
            st.write(f"**Fold {fold + 1}/{n_splits}**")
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            _, predictions, *metrics = self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification)
            scores.append(metrics)

        scores = np.array(scores)

        if is_classification:
            avg_acc, avg_f1 = scores.mean(axis=0)
            st.markdown(f"## Average Metrics after {n_splits}-Fold CV")
            st.markdown(f"<span style='color:green; font-weight:bold;'>Accuracy:</span> {avg_acc:.4f}  \n"
                        f"<span style='color:green; font-weight:bold;'>F1-score:</span> {avg_f1:.4f}", unsafe_allow_html=True)
            return avg_acc, avg_f1
        else:
            avg_r2, avg_mae, avg_mse, avg_rmse = scores.mean(axis=0)
            st.markdown(f"## Average Metrics after {n_splits}-Fold CV")
            st.markdown(f"<span style='color:green; font-weight:bold;'>R²:</span> {avg_r2:.4f}  \n"
                        f"<span style='color:green; font-weight:bold;'>MAE:</span> {avg_mae:.4f}  \n"
                        f"<span style='color:green; font-weight:bold;'>MSE:</span> {avg_mse:.4f}  \n"
                        f"<span style='color:green; font-weight:bold;'>RMSE:</span> {avg_rmse:.4f}", unsafe_allow_html=True)
            return avg_r2, avg_mae, avg_mse, avg_rmse
    
    def is_single_output(self, y):
        return len(y.shape) == 1 or y.shape[1] == 1

    def wrap_regressor_if_needed(self, model, y):
        if self.is_single_output(y):
            return model
        else:
            return MultiOutputRegressor(model)
        
    def wrap_classifier_if_needed(self, model, y):
        if self.is_single_output(y):
            return model
        else:
            return MultiOutputClassifier(model)
    
    ## Regression Model ###

    def Linear_regressor(self, X_train, X_test, y_train, y_test):
        #fit_intercept = self.get_param("Fit intercept (True/False)", True, lambda x: x.lower() == 'true')
        fit_intercept_option = st.selectbox("Fit Intercept:", ["True", "False"], index=0)
        fit_intercept = fit_intercept_option == "True"
        model = LinearRegression(fit_intercept=fit_intercept)
        model = self.wrap_regressor_if_needed(model, y_train)
        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification=False)

    def Ridge_regression(self, X_train, wrap_regressor_if_needed ,X_test, y_train, y_test):
        alpha = st.selectbox("Alpha", 1.0, float)
        solver = st.selectbox("Solver (auto/svd/cholesky/lsqr/sparse_cg/sag/saga)", "auto", str)
        max_iter = st.selectbox("Max iterations", None, lambda x: int(x) if x.strip() else None)
        tol = st.selectbox("Tolerance", 0.001, float)
        model = Ridge(alpha=alpha, solver=solver, max_iter=max_iter, tol=tol)
        model = self.wrap_regressor_if_needed(model, y_train)
        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification=False)

    def ElasticNet_regression(self, X_train, X_test, y_train, y_test):
        alpha = self.get_param("Alpha", 1.0, float)
        l1_ratio = self.get_param("L1 ratio (0-1)", 0.5, float)
        max_iter = self.get_param("Max iterations", 1000, int)
        tol = self.get_param("Tolerance", 0.0001, float)
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, tol=tol)
        model = self.wrap_regressor_if_needed(model, y_train)
        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification=False)

    def Lasso_regression(self, X_train, X_test, y_train, y_test):
        alpha = self.get_param("Alpha", 1.0, float)
        max_iter = self.get_param("Max iterations", 1000, int)
        tol = self.get_param("Tolerance", 0.0001, float)
        model = Lasso(alpha=alpha, max_iter=max_iter, tol=tol)
        model = self.wrap_regressor_if_needed(model, y_train)
        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification=False)

    def Decision_tree_regression(self, X_train, X_test, y_train, y_test):
        max_depth = self.get_param("Max depth", None, lambda x: int(x) if x.strip() else None)
        min_samples_split = self.get_param("Min samples split", 2, int)
        min_samples_leaf = self.get_param("Min samples leaf", 1, int)
        model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
        model = self.wrap_regressor_if_needed(model, y_train)
        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification=False)

    def Random_forest_regression(self, X_train, X_test, y_train, y_test):
        n_estimators = self.get_param("Number of estimators", 100, int)
        max_depth = self.get_param("Max depth", None, lambda x: int(x) if x.strip() else None)
        min_samples_split = self.get_param("Min samples split", 2, int)
        min_samples_leaf = self.get_param("Min samples leaf", 1, int)
        max_features = self.get_param("Max features (sqrt/log2/None)", "log2", lambda x: None if x.strip().lower() == "none" else x.strip())
        bootstrap = self.get_param("Bootstrap (True/False)", True, lambda x: x.lower() == 'true')
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                                    min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                    max_features=max_features, bootstrap=bootstrap, n_jobs=-1, random_state=42)
        model = self.wrap_regressor_if_needed(model, y_train)
        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification=False)

    def Gradient_boosting_regressor(self, X_train, X_test, y_train, y_test):
        n_estimators = self.get_param("Number of estimators", 100, int)
        learning_rate = self.get_param("Learning rate", 0.1, float)
        max_depth = self.get_param("Max depth", 3, int)
        min_samples_split = self.get_param("Min samples split", 2, int)
        min_samples_leaf = self.get_param("Min samples leaf", 1, int)
        subsample = self.get_param("Subsample (0.5 - 1.0)", 1.0, float)
        max_features = self.get_param("Max features (sqrt/log2/None)", None, lambda x: None if x.strip().lower() == "none" else x.strip())
        model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                                        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                        subsample=subsample, max_features=max_features)
        model = self.wrap_regressor_if_needed(model, y_train)
        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification=False)

    def AdaBoost_regressor(self, X_train, X_test, y_train, y_test):
        n_estimators = self.get_param("Number of estimators", 50, int)
        learning_rate = self.get_param("Learning rate", 1.0, float)
        loss = self.get_param("Loss (linear/square/exponential)", "linear", str)
        model = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate, loss=loss, random_state=42)
        model = self.wrap_regressor_if_needed(model, y_train)
        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification=False)

    def SVR_regression(self, X_train, X_test, y_train, y_test):
        kernel = self.get_param("Kernel (linear/poly/rbf/sigmoid)", "rbf", str)
        C = self.get_param("Regularization parameter C", 1.0, float)
        epsilon = self.get_param("Epsilon in loss function", 0.1, float)
        gamma = self.get_param("Gamma (scale/auto)", "scale", str)
        model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
        model = self.wrap_regressor_if_needed(model, y_train)
        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification=False)

    def KNN_regressor(self, X_train, X_test, y_train, y_test):
        n_neighbors = self.get_param("Number of neighbors", 5, int)
        weights = self.get_param("Weights (uniform/distance)", "uniform", str)
        algorithm = self.get_param("Algorithm (auto/ball_tree/kd_tree/brute)", "auto", str)
        model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
        model = self.wrap_regressor_if_needed(model, y_train)
        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification=False)

    def Xgb_regressor(self, X_train, X_test, y_train, y_test):
        n_estimators = self.get_param("Number of estimators", 100, int)
        learning_rate = self.get_param("Learning rate", 0.1, float)
        max_depth = self.get_param("Max depth", 3, int)
        subsample = self.get_param("Subsample", 1.0, float)
        colsample_bytree = self.get_param("Colsample bytree", 1.0, float)
        gamma = self.get_param("Gamma", 0, float)
        reg_alpha = self.get_param("L1 regularization (alpha)", 0, float)
        reg_lambda = self.get_param("L2 regularization (lambda)", 1, float)
        model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                            subsample=subsample, colsample_bytree=colsample_bytree, gamma=gamma,
                            reg_alpha=reg_alpha, reg_lambda=reg_lambda, objective='reg:squarederror')
        model = self.wrap_regressor_if_needed(model, y_train)
        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification=False)
    

    def GaussianProcess_regression(self, X_train, X_test, y_train, y_test):
        st.subheader("Gaussian Process Regressor Settings")

        # Kernel selection
        kernel_option = st.selectbox(
            "Kernel",
            ["RBF", "Matern", "RationalQuadratic", "DotProduct", "WhiteKernel"],
            index=0
        )

        # Map selection to actual kernel object
        kernel_map = {
            "RBF": RBF(),
            "Matern": Matern(nu=1.5),
            "RationalQuadratic": RationalQuadratic(),
            "DotProduct": DotProduct(),
            "WhiteKernel": WhiteKernel()
        }
        kernel = kernel_map[kernel_option]

        # Hyperparameters
        col1, col2, col3 = st.columns(3)

        with col1:
            alpha = st.number_input("Alpha (noise level)", min_value=1e-12, value=1e-10, step=1e-10, format="%.1e")
        
        with col2:
            optimizer = st.selectbox("Optimizer", ["fmin_l_bfgs_b", None], index=0)
        
        with col3:
            n_restarts_optimizer = st.number_input("Restarts for Optimizer", min_value=0, value=0, step=1)

        normalize_y = st.checkbox("Normalize y", value=False)
        copy_X_train = st.checkbox("Copy X_train", value=True)

        # Instantiate and fit model
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y,
            copy_X_train=copy_X_train
        )

        model = self.wrap_regressor_if_needed(model, y_train)
        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification=False)



    
    ## Classifier###

    def Logistic_regression(self, X_train, X_test, y_train, y_test):
        penalty = self.get_param("Penalty (l1, l2, elasticnet, none)", "l2", str)
        C = self.get_param("Inverse regularization strength C", 1.0, float)
        solver = self.get_param("Solver (lbfgs, saga, liblinear)", "lbfgs", str)
        max_iter = self.get_param("Max iterations", 100, int)
        model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=max_iter)
        model = self.wrap_classifier_if_needed(model, y_train)
        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification=True)

    def Decision_tree_classifier(self, X_train, X_test, y_train, y_test):
        max_depth = self.get_param("Max depth", None, lambda x: int(x) if x.strip() else None)
        min_samples_split = self.get_param("Min samples split", 2, int)
        min_samples_leaf = self.get_param("Min samples leaf", 1, int)
        criterion = self.get_param("Criterion (gini, entropy)", "gini", str)
        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf, criterion=criterion)
        model = self.wrap_classifier_if_needed(model, y_train)
        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification=True)

    def Random_forest_classifier(self, X_train, X_test, y_train, y_test):
        n_estimators = self.get_param("Number of estimators", 100, int)
        max_depth = self.get_param("Max depth", None, lambda x: int(x) if x.strip() else None)
        min_samples_split = self.get_param("Min samples split", 2, int)
        min_samples_leaf = self.get_param("Min samples leaf", 1, int)
        criterion = self.get_param("Criterion (gini, entropy)", "gini", str)
        max_features = self.get_param("Max features (sqrt/log2/None)", "auto", lambda x: None if x.strip().lower() == "none" else x.strip())
        bootstrap = self.get_param("Bootstrap (True/False)", True, lambda x: x.lower() == 'true')
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf, criterion=criterion, max_features=max_features,
                                    bootstrap=bootstrap, n_jobs=-1, random_state=42)
        model = self.wrap_classifier_if_needed(model, y_train)
        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification=True)

    def Gradient_boosting_classifier(self, X_train, X_test, y_train, y_test):
        n_estimators = self.get_param("Number of estimators", 100, int)
        learning_rate = self.get_param("Learning rate", 0.1, float)
        max_depth = self.get_param("Max depth", 3, int)
        min_samples_split = self.get_param("Min samples split", 2, int)
        min_samples_leaf = self.get_param("Min samples leaf", 1, int)
        subsample = self.get_param("Subsample (0.5 - 1.0)", 1.0, float)
        max_features = self.get_param("Max features (auto/sqrt/log2/None)", None, lambda x: None if x.strip().lower() == "none" else x.strip())
        model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                                        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                        subsample=subsample, max_features=max_features)
        model = self.wrap_classifier_if_needed(model, y_train)
        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification=True)

    def AdaBoost_classifier(self, X_train, X_test, y_train, y_test):
        n_estimators = self.get_param("Number of estimators", 50, int)
        learning_rate = self.get_param("Learning rate", 1.0, float)
        algorithm = self.get_param("Algorithm (SAMME, SAMME.R)", "SAMME.R", str)
        model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, algorithm=algorithm, random_state=42)
        model = self.wrap_classifier_if_needed(model, y_train)
        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification=True)

    def SVC_classifier(self, X_train, X_test, y_train, y_test):
        kernel = self.get_param("Kernel (linear, poly, rbf, sigmoid)", "rbf", str)
        C = self.get_param("Regularization parameter C", 1.0, float)
        gamma = self.get_param("Gamma (scale, auto)", "scale", str)
        probability = self.get_param("Probability estimates (True/False)", False, lambda x: x.lower() == 'true')
        model = SVC(kernel=kernel, C=C, gamma=gamma, probability=probability)
        model = self.wrap_classifier_if_needed(model, y_train)
        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification=True)

    def KNN_classifier(self, X_train, X_test, y_train, y_test):
        n_neighbors = self.get_param("Number of neighbors", 5, int)
        weights = self.get_param("Weights (uniform/distance)", "uniform", str)
        algorithm = self.get_param("Algorithm (auto/ball_tree/kd_tree/brute)", "auto", str)
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
        model = self.wrap_classifier_if_needed(model, y_train)
        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification=True)

    def XGBoost_classifier(self, X_train, X_test, y_train, y_test):
        n_estimators = self.get_param("Number of estimators", 100, int)
        learning_rate = self.get_param("Learning rate", 0.1, float)
        max_depth = self.get_param("Max depth", 3, int)
        subsample = self.get_param("Subsample", 1.0, float)
        colsample_bytree = self.get_param("Colsample bytree", 1.0, float)
        gamma = self.get_param("Gamma", 0, float)
        reg_alpha = self.get_param("L1 regularization (alpha)", 0, float)
        reg_lambda = self.get_param("L2 regularization (lambda)", 1, float)
        model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                            subsample=subsample, colsample_bytree=colsample_bytree, gamma=gamma,
                            reg_alpha=reg_alpha, reg_lambda=reg_lambda, use_label_encoder=False, eval_metric='logloss')
        model = self.wrap_classifier_if_needed(model, y_train)
        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification=True)


    
    '''def save_predictions_to_csv(self, y_test, predictions, filename="predictions.csv"):
        
        output_df = pd.DataFrame({
            "Actual": y_test.values,
            "Predicted": predictions
        })
        output_df.to_csv(filename, index=False)
        print(f"Predictions saved to {filename}")


'''
    def save_predictions_to_csv(self, y_test, predictions, filename="predictions.csv"):
        y_test = np.array(y_test)
        predictions = np.array(predictions)

        if y_test.ndim == 2 and y_test.shape[1] == 1:
            y_test = y_test.ravel()
        if predictions.ndim == 2 and predictions.shape[1] == 1:
            predictions = predictions.ravel()

        if y_test.ndim == 2 and predictions.ndim == 2:
            columns = [f"Actual_T{i}" for i in range(y_test.shape[1])] + [f"Predicted_T{i}" for i in range(predictions.shape[1])]
            data = np.hstack((y_test, predictions))
            output_df = pd.DataFrame(data, columns=columns)
        else:
            output_df = pd.DataFrame({
                "Actual": y_test,
                "Predicted": predictions
            })

        output_df.to_csv(filename, index=False)


class optuna_Model:
    def __init__(self , n_trials = 100):
        self.model = None
        self.predictions = None
        self.study = None
        self.n_trials = n_trials
    
    
    def fit_predict_evaluate(self, model, X_train, X_test, y_train, y_test, study, is_classification=True):
        model.fit(X_train, y_train)
        self.predictions = model.predict(X_test)
        self.model = model

        if is_classification:
            acc = accuracy_score(y_test, self.predictions)
            f1 = f1_score(y_test, self.predictions, average="weighted")
            return model, self.predictions, acc, f1
        else:
            r2 = r2_score(y_test, self.predictions)
            mae = mean_absolute_error(y_test, self.predictions)
            mse = mean_squared_error(y_test, self.predictions)
            rmse = np.sqrt(mse)
            return model, self.predictions, r2, mae, mse, rmse , study 
        
    def save_predictions_to_csv(self, y_test, predictions, filename="predictions.csv"):
        y_test = np.array(y_test)
        predictions = np.array(predictions)

        if y_test.ndim == 2 and y_test.shape[1] == 1:
            y_test = y_test.ravel()
        if predictions.ndim == 2 and predictions.shape[1] == 1:
            predictions = predictions.ravel()

        if y_test.ndim == 2 and predictions.ndim == 2:
            columns = [f"Actual_T{i}" for i in range(y_test.shape[1])] + [f"Predicted_T{i}" for i in range(predictions.shape[1])]
            data = np.hstack((y_test, predictions))
            output_df = pd.DataFrame(data, columns=columns)
        else:
            output_df = pd.DataFrame({
                "Actual": y_test,
                "Predicted": predictions
            })

        output_df.to_csv(filename, index=False)

    def is_single_output(self, y):
        return len(y.shape) == 1 or y.shape[1] == 1

    def wrap_regressor_if_needed(self, model, y):
        if self.is_single_output(y):
            return model
        else:
            return MultiOutputRegressor(model)
        
    def wrap_classifier_if_needed(self, model, y):
        if self.is_single_output(y):
            return model
        else:
            return MultiOutputClassifier(model)

    ############################################################## REGRESSION MODEL ########################################################################################

    def Linear_regressor(self, X_train, X_test, y_train, y_test):

        model = LinearRegression()
        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification=False)

    def Ridge_regression(self, X_train, X_test, y_train, y_test):

        def objective(trial):
            alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
            solver = trial.suggest_categorical("solver", ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"])
            max_iter = trial.suggest_int("max_iter", 100, 10000)
            tol = trial.suggest_float("tol", 1e-5, 1e-1, log=True)

            model = Ridge(alpha=alpha, solver=solver, max_iter=max_iter, tol=tol)
            model = self.wrap_regressor_if_needed(model, y_train)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            return mean_squared_error(y_test, pred)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)
        self.study = study
        best_params = study.best_params

        model = Ridge(**best_params)
        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, study ,  is_classification=False)

    def ElasticNet_regression(self, X_train, X_test, y_train, y_test):

        def objective(trial):
            alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
            l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
            max_iter = trial.suggest_int("max_iter", 100, 10000)
            tol = trial.suggest_float("tol", 1e-5, 1e-1, log=True)

            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, tol=tol)
            model = self.wrap_regressor_if_needed(model, y_train)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            return mean_squared_error(y_test, pred)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)
        self.study = study
        best_params = study.best_params

        model = ElasticNet(**best_params)
        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, study ,  is_classification=False)

    def Lasso_regression(self, X_train, X_test, y_train, y_test):

        def objective(trial):
            alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
            fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
            model = Lasso(alpha=alpha, fit_intercept=fit_intercept, max_iter=10000)
            model = self.wrap_regressor_if_needed(model, y_train)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            return mean_squared_error(y_test, pred)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)
        best_params = study.best_params
        self.study = study

        model = Lasso(**best_params, max_iter=10000)
        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, study ,  is_classification=False)

    def Decision_tree_regression(self, X_train, X_test, y_train, y_test):

        def objective(trial):
            max_depth = trial.suggest_int("max_depth", 1, 30)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
            max_features = trial.suggest_categorical("max_features", [ "sqrt", "log2", None])
            model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features
            )
            model = self.wrap_regressor_if_needed(model, y_train)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            return mean_squared_error(y_test, pred)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)
        best_params = study.best_params
        self.study = study

        model = DecisionTreeRegressor(**best_params)
        model = self.wrap_regressor_if_needed(model, y_train)
        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, study ,  is_classification=False)

    def Random_forest_regression(self, X_train, X_test, y_train, y_test):

        def objective(trial):
            n_estimators = trial.suggest_int("n_estimators", 50, 300)
            max_depth = trial.suggest_int("max_depth", 2, 15)  # Lower max depth
            min_samples_split = trial.suggest_int("min_samples_split", 5, 20)  
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 3, 20)
            max_features = trial.suggest_categorical("max_features", [ "sqrt", "log2"])
            bootstrap = trial.suggest_categorical("bootstrap", [True, False])

            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                bootstrap=bootstrap,
                n_jobs=-1
            )
            model = self.wrap_regressor_if_needed(model , y_train)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            return mean_squared_error(y_test, pred)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)
        best_params = study.best_params
        self.study = study

        model = RandomForestRegressor(**best_params, n_jobs=-1)
        model = self.wrap_regressor_if_needed(model, y_train)
        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, study ,  is_classification=False)

    def Gradient_boosting_regressor(self, X_train, X_test, y_train, y_test):

        def objective(trial):
            n_estimators = trial.suggest_int("n_estimators", 100, 1000)  # Increased from 50-100
            learning_rate = trial.suggest_float("learning_rate", 0.005, 0.3, log=True)  # Expanded range
            max_depth = trial.suggest_int("max_depth", 3, 12)  # Slightly reduced max to prevent overfitting
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 15)  # Reduced max
            subsample = trial.suggest_float("subsample", 0.6, 1.0)  # Increased minimum
            max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None, 0.3, 0.5, 0.7])  # Added float options
            
            # ADDITIONAL REGULARIZATION PARAMETERS
            alpha = trial.suggest_float("alpha", 0.0, 0.5)  # L1 regularization
            
            # EARLY STOPPING PARAMETERS
            validation_fraction = trial.suggest_float("validation_fraction", 0.1, 0.3)
            n_iter_no_change = trial.suggest_int("n_iter_no_change", 5, 20)
            tol = trial.suggest_float("tol", 1e-6, 1e-3, log=True)
        
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                subsample=subsample,
                max_features=max_features,
                alpha=alpha,
                validation_fraction=validation_fraction,
                n_iter_no_change=n_iter_no_change,
                tol=tol,
                random_state=42
            )
            model = self.wrap_regressor_if_needed(model , y_train)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            return mean_squared_error(y_test, pred)

        study = optuna.create_study(
                    direction="minimize",
                    sampler=optuna.samplers.TPESampler(seed=42), 
                    pruner=optuna.pruners.MedianPruner()  
                )        
        study.optimize(objective, n_trials=self.n_trials , n_jobs = -1 )
        best_params = study.best_params
        self.study = study

        model = GradientBoostingRegressor(**best_params)
        model = self.wrap_regressor_if_needed(model, y_train)
        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, study ,  is_classification=False)

    def AdaBoost_regressor(self, X_train, X_test, y_train, y_test):

        def objective(trial):
            n_estimators = trial.suggest_int("n_estimators", 50, 300)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 2.0, log=True)
            loss = trial.suggest_categorical("loss", ["linear", "square", "exponential"])

            model = AdaBoostRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                loss=loss
            )
            model = self.wrap_regressor_if_needed(model , y_train)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            return mean_squared_error(y_test, pred)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials ,  n_jobs = -1)
        best_params = study.best_params
        self.study = study

        model = AdaBoostRegressor(**best_params)
        model = self.wrap_regressor_if_needed(model, y_train)

        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, study ,  is_classification=False)
    
    
    def Xgb_regressor(self, X_train, X_test, y_train, y_test):

        def objective(trial):
            n_estimators = trial.suggest_int("n_estimators", 50, 300)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.15)
            max_depth = trial.suggest_int("max_depth", 2, 6)
            min_child_weight = trial.suggest_int("min_child_weight", 3, 15)
            subsample = trial.suggest_float("subsample", 0.6, 0.9)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 0.9)
            gamma = trial.suggest_float("gamma", 0.1, 2.0)
            reg_alpha = trial.suggest_float("reg_alpha", 0.1, 10)
            reg_lambda = trial.suggest_float("reg_lambda", 1, 20)

            model = XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                gamma=gamma,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=42,
                objective='reg:squarederror'
            )

            model = self.wrap_regressor_if_needed(model, y_train)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            return mean_squared_error(y_test, pred)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials , n_jobs=-1 )
        best_params = study.best_params
        self.study = study

        model = XGBRegressor(**best_params, objective='reg:squarederror', random_state=42)
        model = self.wrap_regressor_if_needed(model, y_train)

        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, study ,  is_classification=False)

            
    def SVR_regression(self, X_train, X_test, y_train, y_test):
        def objective(trial):
            kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
            C = trial.suggest_float("C", 1e-2, 100.0, log=True)
            epsilon = trial.suggest_float("epsilon", 1e-2, 0.5, log=True)
            gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
            
            model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
            model = self.wrap_regressor_if_needed(model, y_train)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            return mean_squared_error(y_test, preds)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)
        best_params = study.best_params
        self.study = study

        model = SVR(**best_params)
        model = self.wrap_regressor_if_needed(model, y_train)
        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, study ,  is_classification=False)
    
    def KNN_regressor(self, X_train, X_test, y_train, y_test):

        def objective(trial):
            n_neighbors = trial.suggest_int("n_neighbors", 1, 50)
            weights = trial.suggest_categorical("weights", ["uniform", "distance"])
            algorithm = trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"])
            leaf_size = trial.suggest_int("leaf_size", 10, 100)
            p = trial.suggest_int("p", 1, 2)  # 1 = manhattan, 2 = euclidean

            model = KNeighborsRegressor(
                n_neighbors=n_neighbors,
                weights=weights,
                algorithm=algorithm,
                leaf_size=leaf_size,
                p=p
            )
            model = self.wrap_regressor_if_needed(model, y_train)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            return mean_squared_error(y_test, preds)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)
        best_params = study.best_params
        self.study = study

        model = KNeighborsRegressor(**best_params)
        model = self.wrap_regressor_if_needed(model, y_train)
        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, study ,  is_classification=False)

    def GaussianProcess_regression(self, X_train, X_test, y_train, y_test):
        # st.subheader(" Gaussian Process Regression with Optuna Optimization")

        # n_trials = st.number_input("Number of Optuna Trials", min_value=10, max_value=200, value=30, step=5)

        def objective(trial):
            kernel_choice = trial.suggest_categorical("kernel", ["RBF", "Matern", "RationalQuadratic", "DotProduct", "WhiteKernel"])
            kernel_map = {
                "RBF": RBF(),
                "Matern": Matern(nu=1.5),
                "RationalQuadratic": RationalQuadratic(),
                "DotProduct": DotProduct(),
                "WhiteKernel": WhiteKernel()
            }
            kernel = kernel_map[kernel_choice]

            alpha = trial.suggest_float("alpha", 0.05, 1.0, log=True)
            normalize_y = trial.suggest_categorical("normalize_y", [True, False])
            n_restarts_optimizer = trial.suggest_int("n_restarts_optimizer", 0, 5)
            copy_X_train = trial.suggest_categorical("copy_X_train", [True, False])
            optimizer = trial.suggest_categorical("optimizer", ["fmin_l_bfgs_b", None])

            model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=alpha,
                normalize_y=normalize_y,
                n_restarts_optimizer=n_restarts_optimizer,
                copy_X_train=copy_X_train,
                optimizer=optimizer
            )

            model = self.wrap_regressor_if_needed(model, y_train)

            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                mse = mean_squared_error(y_test, preds)
                return mse
            except Exception:
                return float("inf")  
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)

        self.study = study
        best_params = study.best_params


        # Reconstruct best model from optimized params
        kernel = {
            "RBF": RBF(),
            "Matern": Matern(nu=1.5),
            "RationalQuadratic": RationalQuadratic(),
            "DotProduct": DotProduct(),
            "WhiteKernel": WhiteKernel()
        }[best_params["kernel"]]

        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=best_params["alpha"],
            normalize_y=best_params["normalize_y"],
            n_restarts_optimizer=best_params["n_restarts_optimizer"],
            copy_X_train=best_params["copy_X_train"],
            optimizer=best_params["optimizer"]
        )

        model.fit(X_train, y_train)

        # Evaluate on train and test
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        train_r2 = r2_score(y_train, train_preds)
        test_r2 = r2_score(y_test, test_preds)
        train_rmse =math.sqrt( mean_squared_error(y_train, train_preds))
        test_rmse = math.sqrt(mean_squared_error(y_test, test_preds))

        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, study ,  is_classification=False)
    
    def PLS_regression(self, X_train, X_test, y_train, y_test):
        
        def objective(trial):
            n_components = trial.suggest_int("n_components", 1, min(X_train.shape[1], 42))
            scale = trial.suggest_categorical("scale", [True, False])
            
            model = PLSRegression(n_components=n_components, scale=scale)
            model = self.wrap_regressor_if_needed(model, y_train)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            return mean_squared_error(y_test, pred)
        
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials  )
        best_params = study.best_params
        self.study = study
        
        model = PLSRegression(**best_params)
        model = self.wrap_regressor_if_needed(model, y_train)
        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, study ,  is_classification=False)
    
    def Ensemble_regressor(self, X_train, X_test, y_train, y_test):
        def objective(trial):
            ensemble_method = trial.suggest_categorical("ensemble_method", ["voting", "stacking"])
            
            use_svr = trial.suggest_categorical("use_svr", [True, False])
            use_rf = trial.suggest_categorical("use_rf", [True, False])
            use_pls = trial.suggest_categorical("use_pls", [True, False])
            use_ridge = trial.suggest_categorical("use_ridge", [True, False])
            use_xgb = trial.suggest_categorical("use_xgb", [True, False])
            
            model_flags = [use_svr, use_rf, use_pls, use_ridge, use_xgb]
            if sum(model_flags) < 2:
                return float('inf')
            
            base_models = []
            
            if use_svr:
                svr_C = trial.suggest_float("svr_C", 0.1, 100.0, log=True)
                svr_kernel = trial.suggest_categorical("svr_kernel", ["rbf", "linear", "poly"])
                svr_epsilon = trial.suggest_float("svr_epsilon", 0.01, 1.0, log=True)
                
                svr_model = SVR(C=svr_C, kernel=svr_kernel, epsilon=svr_epsilon, gamma='scale')
                base_models.append(('svr', svr_model))
            
            if use_rf:
                rf_n_estimators = trial.suggest_int("rf_n_estimators", 50, 200)
                rf_max_depth = trial.suggest_int("rf_max_depth", 3, 15)
                rf_min_samples_leaf = trial.suggest_int("rf_min_samples_leaf", 2, 10)
                
                rf_model = RandomForestRegressor(
                    n_estimators=rf_n_estimators,
                    max_depth=rf_max_depth,
                    min_samples_leaf=rf_min_samples_leaf,
                    random_state=42,
                    n_jobs=-1
                )
                base_models.append(('rf', rf_model))
            
            if use_pls:
                pls_n_components = trial.suggest_int("pls_n_components", 2, min(X_train.shape[1], 30))
                pls_model = PLSRegression(n_components=pls_n_components, scale=True)
                base_models.append(('pls', pls_model))
            
            if use_ridge:
                ridge_alpha = trial.suggest_float("ridge_alpha", 0.1, 100.0, log=True)
                ridge_model = Ridge(alpha=ridge_alpha)
                base_models.append(('ridge', ridge_model))
            
            if use_xgb:
                xgb_n_estimators = trial.suggest_int("xgb_n_estimators", 50, 150)
                xgb_max_depth = trial.suggest_int("xgb_max_depth", 2, 6)
                xgb_learning_rate = trial.suggest_float("xgb_learning_rate", 0.01, 0.15)
                xgb_reg_lambda = trial.suggest_float("xgb_reg_lambda", 1, 10)
                
                xgb_model = XGBRegressor(
                    n_estimators=xgb_n_estimators,
                    max_depth=xgb_max_depth,
                    learning_rate=xgb_learning_rate,
                    reg_lambda=xgb_reg_lambda,
                    random_state=42
                )
                base_models.append(('xgb', xgb_model))
            
            if ensemble_method == "voting":
                ensemble_model = VotingRegressor(estimators=base_models)
            else:  
                meta_learner = Ridge(alpha=trial.suggest_float("meta_alpha", 0.1, 10.0))
                cv_folds = trial.suggest_int("stacking_cv", 3, 7)
                
                ensemble_model = StackingRegressor(
                    estimators=base_models,
                    final_estimator=meta_learner,
                    cv=cv_folds,
                    n_jobs=-1
                )
            
            ensemble_model = self.wrap_regressor_if_needed(ensemble_model, y_train)
            
            try:
                cv_scores = cross_val_score(
                    ensemble_model, X_train, y_train,
                    cv=5,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
                return -cv_scores.mean()
            except Exception as e:
                return float('inf')
        
        study = optuna.create_study(direction="minimize")
        
        # with st.spinner("Optimizing Ensemble Regressor..."):
        study.optimize(objective, n_trials=self.n_trials )
        
        best_params = study.best_params
        self.study = study
        
        
        # Rebuild best ensemble model
        base_models = []
        
        if best_params.get("use_svr", False):
            svr_model = SVR(
                C=best_params["svr_C"],
                kernel=best_params["svr_kernel"],
                epsilon=best_params["svr_epsilon"],
                gamma='scale'
            )
            base_models.append(('svr', svr_model))
        
        if best_params.get("use_rf", False):
            rf_model = RandomForestRegressor(
                n_estimators=best_params["rf_n_estimators"],
                max_depth=best_params["rf_max_depth"],
                min_samples_leaf=best_params["rf_min_samples_leaf"],
                random_state=42,
                n_jobs=-1
            )
            base_models.append(('rf', rf_model))
        
        if best_params.get("use_pls", False):
            pls_model = PLSRegression(
                n_components=best_params["pls_n_components"],
                scale=True
            )
            base_models.append(('pls', pls_model))
            
        if best_params.get("use_ridge", False):
            ridge_model = Ridge(alpha=best_params["ridge_alpha"])
            base_models.append(('ridge', ridge_model))
        
        if best_params.get("use_xgb", False):
            xgb_model = XGBRegressor(
                n_estimators=best_params["xgb_n_estimators"],
                max_depth=best_params["xgb_max_depth"],
                learning_rate=best_params["xgb_learning_rate"],
                reg_lambda=best_params["xgb_reg_lambda"],
                random_state=42
            )
            base_models.append(('xgb', xgb_model))
        
        if best_params["ensemble_method"] == "voting":
            final_ensemble = VotingRegressor(estimators=base_models)
        else:
            meta_learner = Ridge(alpha=best_params["meta_alpha"])
            final_ensemble = StackingRegressor(
                estimators=base_models,
                final_estimator=meta_learner,
                cv=best_params["stacking_cv"],
                n_jobs=-1
            )
        
        final_ensemble = self.wrap_regressor_if_needed(final_ensemble, y_train)
        
        return self.fit_predict_evaluate(final_ensemble, X_train, X_test, y_train, y_test, study , is_classification=False)
    
    def ANN_regression(self, X_train, X_test, y_train, y_test):
    
        def create_ann_model(n_layers, n_neurons, activation, learning_rate, dropout_rate, optimizer_name):
            model = Sequential()
            
            # Input layer
            model.add(Dense(n_neurons, activation=activation, input_shape=(X_train.shape[1],)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
            
            # Hidden layers
            for _ in range(n_layers - 1):
                model.add(Dense(n_neurons, activation=activation))
                # model.add(BatchNormalization())
                model.add(Dropout(dropout_rate))
            
            # Output layer for regression
            model.add(Dense(1 if len(y_train.shape) == 1 else y_train.shape[1], activation='linear'))
            
            # Compile model
            if optimizer_name == 'adam':
                optimizer = Adam(learning_rate=learning_rate)
            elif optimizer_name == 'rmsprop':
                optimizer = RMSprop(learning_rate=learning_rate)
            else:
                optimizer = SGD(learning_rate=learning_rate)
                
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            return model
        
        def objective(trial):
            # Hyperparameter suggestions
            n_layers = trial.suggest_int("n_layers", 1, 2)
            n_neurons = trial.suggest_int("n_neurons", 16, 64)
            activation = trial.suggest_categorical("activation", ['relu', 'tanh', 'elu'])
            learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True)
            dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.2)
            optimizer_name = trial.suggest_categorical("optimizer", ['adam', 'rmsprop', 'sgd'])
            batch_size = trial.suggest_int("batch_size", 16, 128)
            
            # Create model
            model = create_ann_model(n_layers, n_neurons, activation, learning_rate, dropout_rate, optimizer_name)
            
            # Early stopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            # Fit model
            model.fit(X_train, y_train, 
                    validation_split=0.2, 
                    epochs=100, 
                    batch_size=batch_size,
                    callbacks=[early_stopping],
                    verbose=0)
            
            pred = model.predict(X_test, verbose=0)
            return mean_squared_error(y_test, pred)
        
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials )
        best_params = study.best_params
        self.study = study
        
        # Create best model
        model = create_ann_model(
            best_params['n_layers'],
            best_params['n_neurons'],
            best_params['activation'],
            best_params['learning_rate'],
            best_params['dropout_rate'],
            best_params['optimizer']
        )
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train,
                validation_split=0.2,
                epochs=100,
                batch_size=best_params['batch_size'],
                callbacks=[early_stopping],
                verbose=0)
        
        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, study ,  is_classification=False)


    ## Classification models ###

    def Logistic_regression(self, X_train, X_test, y_train, y_test):
        def objective(trial):
            max_iter = trial.suggest_int("max_iter", 100, 3000)
            C = trial.suggest_float("C", 1e-4, 10.0, log=True)
            penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
            solver = trial.suggest_categorical("solver", ["saga", "liblinear", "lbfgs", "newton-cg", "sag"])
            
            if penalty == "l1" and solver not in ["saga", "liblinear"]:
                return float("inf")
            if penalty == "elasticnet" and solver != "saga":
                return float("inf")
            if penalty == "none" and solver not in ["newton-cg", "lbfgs", "sag"]:
                return float("inf")

            model = LogisticRegression(
                max_iter=max_iter,
                C=C,
                penalty=penalty,
                solver=solver,
                l1_ratio=trial.suggest_float("l1_ratio", 0.0, 1.0) if penalty == "elasticnet" else None,
                random_state=42
            )
            model = self.wrap_classifier_if_needed(model, y_train)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            return 1 - accuracy_score(y_test, pred)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)
        best_params = study.best_params
        self.study = study

        # Filter out l1_ratio if not elasticnet
        if best_params["penalty"] != "elasticnet":
            best_params.pop("l1_ratio", None)

        model = LogisticRegression(**best_params, random_state=42)
        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, study ,  is_classification=True)

    def Decision_tree_classifier(self, X_train, X_test, y_train, y_test):
        def objective(trial):
            max_depth = trial.suggest_int("max_depth", 1, 30)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
            max_features = trial.suggest_categorical("max_features", ["auto", "sqrt", "log2", None])

            model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features
            )
            model = self.wrap_classifier_if_needed(model , y_train)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            return 1 - accuracy_score(y_test, pred)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)
        best_params = study.best_params
        self.study = study

        model = DecisionTreeClassifier(**best_params)
        model = self.wrap_classifier_if_needed(model, y_train)
        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification=True)

    def Random_forest_classifier(self, X_train, X_test, y_train, y_test):
        def objective(trial):
            n_estimators = trial.suggest_int("n_estimators", 50, 300)
            max_depth = trial.suggest_int("max_depth", 5, 30)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
            max_features = trial.suggest_categorical("max_features", ["auto", "sqrt", "log2"])
            bootstrap = trial.suggest_categorical("bootstrap", [True, False])

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                bootstrap=bootstrap,
                n_jobs=-1,
                random_state=42
            )
            model = self.wrap_classifier_if_needed(model , y_train)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            return 1 - accuracy_score(y_test, pred)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)
        best_params = study.best_params
        self.study = study

        model = RandomForestClassifier(**best_params, n_jobs=-1, random_state=42)
        model = self.wrap_classifier_if_needed(model, y_train)
        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification=True)

    def Gradient_boosting_classifier(self, X_train, X_test, y_train, y_test):
        def objective(trial):
            n_estimators = trial.suggest_int("n_estimators", 50, 300)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            max_depth = trial.suggest_int("max_depth", 3, 15)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
            subsample = trial.suggest_float("subsample", 0.5, 1.0)
            max_features = trial.suggest_categorical("max_features", ["auto", "sqrt", "log2", None])

            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                subsample=subsample,
                max_features=max_features
            )
            model = self.wrap_classifier_if_needed(model , y_train)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            return 1 - accuracy_score(y_test, pred)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)
        best_params = study.best_params
        self.study = study

        model = GradientBoostingClassifier(**best_params)
        model = self.wrap_classifier_if_needed(model, y_train)
        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification=True)

    def AdaBoost_classifier(self, X_train, X_test, y_train, y_test):
        def objective(trial):
            n_estimators = trial.suggest_int("n_estimators", 50, 300)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 2.0, log=True)

            model = AdaBoostClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=42
            )
            model = self.wrap_classifier_if_needed(model , y_train)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            return 1 - accuracy_score(y_test, pred)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trails , n_jobs=5)
        best_params = study.best_params
        self.study = study

        model = AdaBoostClassifier(**best_params, random_state=42)
        model = self.wrap_classifier_if_needed(model, y_train)
        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification=True)

    def XGBoost_classifier(self, X_train, X_test, y_train, y_test):
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
                "use_label_encoder": False,
                "eval_metric": "logloss",
                "verbosity": 0,
            }

            model = XGBClassifier(**params)
            model = self.wrap_classifier_if_needed(model, y_train)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            return 1.0 - f1_score(y_test, preds, average="weighted")

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)
        best_params = study.best_params
        self.study = study

        best_params["use_label_encoder"] = False
        best_params["eval_metric"] = "logloss"
        best_params["verbosity"] = 0

        model = XGBClassifier(**best_params)
        model = self.wrap_classifier_if_needed(model, y_train)
        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test,study , is_classification=True)

    def SVR_classifier(self, X_train, X_test, y_train, y_test):
        def objective(trial):
            kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
            C = trial.suggest_float("C", 1e-2, 100.0, log=True)
            gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
            
            model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
            model = self.wrap_classifier_if_needed(model, y_train)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            return 1.0 - f1_score(y_test, preds, average="weighted")

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)
        best_params = study.best_params
        self.study = study

        model = SVC(**best_params, probability=True)
        model = self.wrap_classifier_if_needed(model, y_train)
        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification=True)
    
    def KNN_classifier(self, X_train, X_test, y_train, y_test):
        def objective(trial):
            n_neighbors = trial.suggest_int("n_neighbors", 1, 50)
            weights = trial.suggest_categorical("weights", ["uniform", "distance"])
            algorithm = trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"])
            leaf_size = trial.suggest_int("leaf_size", 10, 100)
            p = trial.suggest_int("p", 1, 2)  # 1 = manhattan, 2 = euclidean

            model = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                algorithm=algorithm,
                leaf_size=leaf_size,
                p=p
            )
            model = self.wrap_classifier_if_needed(model, y_train)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            return 1.0 - f1_score(y_test, preds, average="weighted")

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)
        best_params = study.best_params
        self.study = study

        model = KNeighborsClassifier(**best_params)
        model = self.wrap_classifier_if_needed(model, y_train)
        return self.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification=True)

    def optuna_visualization(self ):
        if not hasattr(self, "study"):
            st.warning("No study available. Run a model training method with Optuna first.")
            return

        st.subheader(" Optimization History")
        fig1 = vis.plot_optimization_history(self.study)
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader(" Hyperparameter Importance")
        fig2 = vis.plot_param_importances(self.study)
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader(" Parallel Coordinate Plot")
        fig3 = vis.plot_parallel_coordinate(self.study)
        st.plotly_chart(fig3, use_container_width=True)
    
class AutoModelSelector:
    def __init__(self, task="regression", scoring_fn=None, verbose=True):
        self.task = task
        self.model_class = optuna_Model()
        self.best_model_name = None
        self.best_score = -np.inf
        self.best_model = None
        self.all_models_scores = []
        self.scoring_fn = scoring_fn
        self.verbose = verbose

    def select_best_model(self, X_train, X_test, y_train, y_test):
        self.all_models_scores = []

        model_methods = {
            "regression": {
                "linear_regressor": self.model_class.Linear_regressor,
                "lasso_regression": self.model_class.Lasso_regression,
                "decision_tree_regression": self.model_class.Decision_tree_regression,
                "random_forest_regression": self.model_class.Random_forest_regression,
                "gradient_boosting_regressor": self.model_class.Gradient_boosting_regressor,
                "adaboost_regressor": self.model_class.AdaBoost_regressor,
                "ridge_regressor": self.model_class.Ridge_regression , 
                "Elastic_Regressor" : self.model_class.ElasticNet_regression , 
                "SVR" : self.model_class.SVR_regression , 
                "Xgb_regressor" : self.model_class.Xgb_regressor,
                "GaussianProcessRegressor" : self.model_class.GaussianProcess_regression,
                "KNN_Regressor":self.model_class.KNN_regressor,
                "Ensemble Regression": self.model_class.Ensemble_regressor , 
                "PLS Regression" : self.model_class.PLS_regression , 
            },
            "classification": {
                "logistic_regression": self.model_class.Logistic_regression,
                "decision_tree_classifier": self.model_class.Decision_tree_classifier,
                "random_forest_classifier": self.model_class.Random_forest_classifier,
                "gradient_boosting_classifier": self.model_class.Gradient_boosting_classifier,
                "adaboost_classifier": self.model_class.AdaBoost_classifier,
                "SVR_classifier" :self.model_class.SVR_classifier ,
                "KNN_classifier" : self.model_class.KNN_classifier , 
                "Xgboost_classifier" : self.model_class.XGBoost_classifier , 
            }
        }

        for name, method in model_methods[self.task].items():
            try:
                if self.verbose:
                    print(f"Training {name}...")
                result = method(X_train, X_test, y_train, y_test)
                model = result[0]
                y_pred = model.predict(X_test)

                if self.scoring_fn:
                    score = self.scoring_fn(y_test, y_pred)
                else:
                    if self.task == "regression":
                        from sklearn.metrics import r2_score
                        score = r2_score(y_test, y_pred)
                    else:
                        from sklearn.metrics import f1_score
                        score = f1_score(y_test, y_pred, average='weighted')

                self.all_models_scores.append((name, score))

                if score > self.best_score:
                    self.best_score = score
                    self.best_model_name = name
                    self.best_model = model
                    print(f' New Best Model: {name} with score: {score:.4f}')

            except Exception as e:
                print(f" {name} failed: {e}")
                self.all_models_scores.append((name, np.nan))  
                continue

        print("\nModel Performance Summary:")
        for name, score in self.all_models_scores:
            print(f"{name:35s}: {score}")

        st.write("\nModel Performance Summary:")
        for name, score in self.all_models_scores:
            st.write(f"{name:35s}: {score}")

        self.plot_model_scores()

        return self.best_model_name, self.best_model, self.best_score, self.all_models_scores

    def plot_model_scores(self):
        names = [name for name, score in self.all_models_scores]
        scores = [score for name, score in self.all_models_scores]

        plt.figure(figsize=(10, 5))
        plt.barh(names, scores, color='skyblue')
        plt.axvline(self.best_score, color='red', linestyle='--', label=f"Best: {self.best_model_name}")
        plt.xlabel("Score")
        plt.title("Model Performance Comparison")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def run_regression(self, X_train, X_test, y_train, y_test):
        self.task = "regression"
        return self.select_best_model(X_train, X_test, y_train, y_test)

    def run_classification(self, X_train, X_test, y_train, y_test):
        self.task = "classification"
        return self.select_best_model(X_train, X_test, y_train, y_test)
    
import numpy as np
import pandas as pd
import pywt

import numpy as np
import pandas as pd
import pywt

class WaveletDenoiser:
    def __init__(self, wavelet='rbio4.4', level=3, threshold_mode='soft'):
        self.wavelet = wavelet
        self.level = level
        self.threshold_mode = threshold_mode
        self.fitted_threshold_ = None
        self.is_fitted_ = False
    
    def _estimate_threshold(self, coeffs):
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(coeffs[-1])))
        return uthresh
    
    def fit(self, X_train):
        if hasattr(X_train, 'values'):
            X_train = X_train.values
        
        thresholds = []
        for i in range(min(100, X_train.shape[0])):
            signal = X_train[i] if len(X_train.shape) > 1 else X_train
            coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
            threshold = self._estimate_threshold(coeffs)
            thresholds.append(threshold)
        
        self.fitted_threshold_ = np.median(thresholds)
        self.is_fitted_ = True
        return self
    
    def _denoise_signal_with_fitted_threshold(self, signal):
        if not self.is_fitted_:
            raise ValueError("This WaveletDenoiser instance is not fitted yet. Call 'fit' first.")
        
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
        
        coeffs_thresh = [coeffs[0]]
        for detail in coeffs[1:]:
            coeffs_thresh.append(pywt.threshold(detail, self.fitted_threshold_, mode=self.threshold_mode))
        
        denoised = pywt.waverec(coeffs_thresh, self.wavelet)
        return denoised[:len(signal)]
    
    def denoise_signal(self, signal):
        if self.is_fitted_:
            return self._denoise_signal_with_fitted_threshold(signal)
        else:
            coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
            uthresh = self._estimate_threshold(coeffs)
            
            coeffs_thresh = [coeffs[0]]
            for detail in coeffs[1:]:
                coeffs_thresh.append(pywt.threshold(detail, uthresh, mode=self.threshold_mode))
            
            denoised = pywt.waverec(coeffs_thresh, self.wavelet)
            return denoised[:len(signal)]
    
    def transform(self, X):
        if hasattr(X, 'values'):
            denoised_data = np.array([self._denoise_signal_with_fitted_threshold(row) 
                                    for row in X.values])
            return pd.DataFrame(denoised_data, columns=X.columns, index=X.index)
        else:
            if len(X.shape) == 1:
                return self._denoise_signal_with_fitted_threshold(X)
            else:
                return np.array([self._denoise_signal_with_fitted_threshold(row) for row in X])
    
    def fit_transform(self, X_train):
        return self.fit(X_train).transform(X_train)
    
    def denoise_dataframe(self, df):
        return self.transform(df)
    
    def denoise_train_test(self, X_train, X_test):
        self.fit(X_train)
        X_train_denoised = self.transform(X_train)
        X_test_denoised = self.transform(X_test)
        return X_train_denoised, X_test_denoised

class OutlierRemover:
    def __init__(self, threshold=2.5):
        """
        threshold: Number of standard deviations used to identify outliers.
        """
        self.threshold = threshold
        self.upper = None
        self.lower = None

    def fit(self, X):
        """
        Calculate mean and std bounds for filtering.
        """
        self.mean = X.values.mean(axis=0)
        self.std = X.values.std(axis=0)
        self.upper = self.mean + self.threshold * self.std
        self.lower = self.mean - self.threshold * self.std

    def transform(self, X):
        """
        Remove spectra that have any value outside the [lower, upper] bounds.
        """
        spectra = X.values
        mask = np.all((spectra >= self.lower) & (spectra <= self.upper), axis=1)
        return X[mask].reset_index(drop=True)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


