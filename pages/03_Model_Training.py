import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys
import importlib.util

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Import data_augmentation module
spec = importlib.util.spec_from_file_location("data_augmentation", os.path.join(PARENT_DIR, "data_augmentation.py"))
data_augmentation_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_augmentation_module)
DataAugmentor = data_augmentation_module.DataAugmentor

from midel import ReadingData, Models, optuna_Model, AutoModelSelector, WaveletDenoiser, OutlierRemover
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import SpectralData
import tempfile
import numpy as np
import os
import json
import pickle
from datetime import datetime
import joblib
import re
from pca import DimensionalityReduction 
from opls import OPLS
import matplotlib.pyplot as plt 
from spectra_specific.Mass_spectra import MassSpectralPreprocessingOptimizer
from spectra_specific.NIRSpectra import NIRPreprocessingOptimizer
from spectra_specific.RamanSpectra import RamanPreprocessingOptimizer
from spectra_specific.FTIRSpectra import FTIRPreprocessingOptimizer
from FFT import FFTProcessor
import copy 
from copy import deepcopy


def initialize_session_state():
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'user_name' not in st.session_state:
        st.session_state.user_name = ""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'targets_set' not in st.session_state:
        st.session_state.targets_set = False
    if 'data_split' not in st.session_state:
        st.session_state.data_split = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'skipped_steps' not in st.session_state:
        st.session_state.skipped_steps = set()
    if 'two_part_models' not in st.session_state:
        st.session_state.two_part_models = {}
    if 'augmentation_history' not in st.session_state:
        st.session_state.augmentation_history = []


def get_current_data():
    if ('augmented_X_train' in st.session_state and 
        'augmented_y_train' in st.session_state and
        st.session_state.augmented_X_train is not None and 
        st.session_state.augmented_X_train.size > 0 and
        st.session_state.augmented_y_train is not None and 
        st.session_state.augmented_y_train.size > 0):
        
        X_train = st.session_state.augmented_X_train
        y_train = st.session_state.augmented_y_train
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        
    elif (hasattr(st.session_state, 'X_train') and 
          hasattr(st.session_state, 'X_test') and
          hasattr(st.session_state, 'y_train') and 
          hasattr(st.session_state, 'y_test')):
        
        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test
    else:
        return None, None, None, None
    
    return X_train, X_test, y_train, y_test


def app():
    initialize_session_state()
    
    st.title("Model Training & Evaluation Features")
    
    if st.session_state.step == 1:
        st.header("Step 1: Load Raw Data")
        
        st.subheader("User Information")
        
        if not st.session_state.user_name:
            user_name_input = st.text_input("Enter your name:")
            if st.button("Submit Name"):
                if user_name_input:
                    st.session_state.user_name = user_name_input
                    st.rerun()
                else:
                    st.warning("Please enter your name")
            return
        
        st.success(f"Welcome, {st.session_state.user_name}")
        
        if st.button("Change Name"):
            st.session_state.user_name = ""
            st.rerun()
        
        st.subheader("Data Upload")
        uploaded_file = st.file_uploader("Upload CSV, XLSX or TXT file", type=["csv", "xlsx", "txt"])
        
        if uploaded_file:
            try:
                rd = ReadingData()
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    file_path = tmp_file.name

                data = rd.read_data(file_path)
                st.session_state.original_data = data.copy()
                st.session_state.current_data = data.copy()
                st.session_state.data_loaded = True
                
                st.success("Data loaded successfully")
                st.dataframe(data.head())
                
                os.unlink(file_path)
                
                if st.button("Proceed to Target Selection"):
                    st.session_state.step = 2
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")

    elif st.session_state.step == 2:
        st.header("Step 2: Target Selection")

        if not st.session_state.data_loaded:
            st.error("Please upload data first")
            if st.button("Go back to Data Upload"):
                st.session_state.step = 1
                st.rerun()
            return

        data = st.session_state.current_data.copy()

        st.subheader("Optional: Drop Columns")
        drop_columns = st.multiselect("Select columns to drop (optional)", data.columns)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Drop Selected Columns"):
                if drop_columns:
                    data = data.drop(columns=drop_columns)
                    st.session_state.current_data = data.copy()
                    st.success(f"Dropped columns: {', '.join(drop_columns)}")
                    st.rerun()

        with col2:
            if st.button("Reset Dataset"):
                st.session_state.current_data = st.session_state.original_data.copy()
                st.session_state.targets_set = False
                st.session_state.data_split = False
                
                if 'X_full' in st.session_state:
                    del st.session_state.X_full
                if 'y_full' in st.session_state:
                    del st.session_state.y_full
                if 'current_X' in st.session_state:
                    del st.session_state.current_X
                if 'target_columns' in st.session_state:
                    del st.session_state.target_columns
                st.success("Dataset reset to original")
                st.rerun()

        st.dataframe(data.head())

        target_columns = st.multiselect("Select Target Columns", data.columns)
        if st.button("Set Targets"):
            if target_columns:
                st.session_state.target_columns = target_columns
                st.session_state.targets_set = True
                X = data.drop(columns=target_columns)
                y = data[target_columns]

                st.session_state.X_full = X
                st.session_state.y_full = y
                st.session_state.current_X = X.copy()

                try:
                    x_axis = X.columns.astype(float)
                    st.session_state.x_axis = x_axis
                except ValueError:
                    st.session_state.x_axis = np.arange(X.shape[1])
                    st.warning("Column names cannot be converted to float. Using indices for plotting.")

                st.success(f"Target columns set: {target_columns}")
                st.info(f"Features: {X.shape}, Targets: {y.shape}")
            else:
                st.error("Please select at least one target column")

        if st.session_state.targets_set:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Proceed to Training/Test Split"):
                    st.session_state.step = 3
                    st.rerun()
            with col2:
                if st.button("Skip to Model Training"):
                    st.session_state.skipped_steps.add(3) 
                    st.session_state.step = 4
                    st.rerun()

    elif st.session_state.step == 3:
        st.header("Step 3: Training/Test Split")
        
        if not st.session_state.targets_set:
            st.error("Please set target columns first")
            if st.button("Go back to Target Selection"):
                st.session_state.step = 2
                st.rerun()
            return
        
        X = st.session_state.X_full
        y = st.session_state.y_full
        
        st.write(f"Total samples: {X.shape[0]}")
        st.write(f"Total features: {X.shape[1]}")
        
        test_size = st.slider("Select test size percentage", min_value=10, max_value=50, value=20, step=5)
        random_state = st.number_input("Random state (for reproducibility)", min_value=0, max_value=1000, value=42, step=1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Perform Train/Test Split"):
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, 
                        test_size=test_size/100, 
                        random_state=int(random_state)
                    )
                    
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.data_split = True
                    
                    st.success("Data split completed successfully")
                    st.write(f"Training set: {X_train.shape[0]} samples")
                    st.write(f"Test set: {X_test.shape[0]} samples")
                    
                except Exception as e:
                    st.error(f"Error during train/test split: {str(e)}")
        
        with col2:
            if st.session_state.data_split:
                if st.button("Proceed to Model Training"):
                    st.session_state.step = 4
                    st.rerun()

    elif st.session_state.step == 4:
        st.header("Step 4: Model Training")
        
        if not st.session_state.data_split:
            st.error("Please complete train-test split first")
            if st.button("Go back to Train/Test Split"):
                st.session_state.step = 3
                st.rerun()
            return
        
        X_train, X_test, y_train, y_test = get_current_data()
        
        if X_train is None:
            st.error("No training data available. Please complete previous steps.")
            return
        
        Data_Type = st.radio("Select Data Type", ("Normal Data", "Zero-Inflated Data"))
        
        if Data_Type == "Normal Data":
            
            manual = Models()
            optuna = optuna_Model()
            auto = AutoModelSelector()
            
            if ('augmented_X_train' in st.session_state and 
                'augmented_y_train' in st.session_state and
                st.session_state.augmented_X_train is not None and st.session_state.augmented_X_train.size > 0 and
                st.session_state.augmented_y_train is not None and st.session_state.augmented_y_train.size > 0 ):

                
                X_train = st.session_state.augmented_X_train
                y_train = st.session_state.augmented_y_train
                X_test = st.session_state.X_test
                y_test = st.session_state.y_test
                st.success("Using augmented training data from step 8")
                
            else:
                try:
                    X_train, X_test, y_train, y_test = get_current_data()
                    if X_train is not None:
                        st.info("Using data from get_current_data()")
                except:
                    X_train = None
            
            if X_train is None:
                if (hasattr(st.session_state, 'X_train') and 
                    hasattr(st.session_state, 'X_test') and
                    hasattr(st.session_state, 'y_train') and 
                    hasattr(st.session_state, 'y_test')):
                    
                    X_train = st.session_state.X_train
                    X_test = st.session_state.X_test
                    y_train = st.session_state.y_train
                    y_test = st.session_state.y_test
                    st.warning("Using original (non-augmented) data as fallback")
                else:
                    st.error("No training data available. Please complete previous steps.")
                    return
            
            if hasattr(y_train, 'values'):
                y_train = y_train.values.ravel()
            else:
                y_train = y_train.ravel()
            
            if hasattr(y_test, 'values'):
                y_test = y_test.values.ravel()
            else:
                y_test = y_test.ravel()
            
            st.write(f"**Training data ready:** Train={X_train.shape}, Test={X_test.shape}")
            st.write(f"**Target shapes:** y_train={y_train.shape}, y_test={y_test.shape}")
            
            if ('augmented_X_train' in st.session_state and 
                st.session_state.augmented_X_train is not None and st.session_state.augmented_X_train.size > 0 and 
                X_train.shape[0] == st.session_state.augmented_X_train.shape[0] ):
                
                original_size = st.session_state.X_train.shape[0] if hasattr(st.session_state, 'X_train') else 'Unknown'
                augmented_size = X_train.shape[0]
                st.success(f"**Augmented Training Data Active:** {original_size} → {augmented_size} samples")
                
                if hasattr(st.session_state, 'augmentation_history') and st.session_state.augmentation_history:
                    with st.expander("View Augmentation History"):
                        for i, aug in enumerate(st.session_state.augmentation_history):
                            st.write(f"{i+1}. {aug}")
            
            if X_train.shape[0] != y_train.shape[0] or X_test.shape[0] != y_test.shape[0]:
                st.error(f"Data inconsistency detected!")
                st.error(f"Training: X={X_train.shape[0]} samples, y={y_train.shape[0]} targets")
                st.error(f"Testing: X={X_test.shape[0]} samples, y={y_test.shape[0]} targets")
                return
                
            model_flow = st.radio("Select model run type", ("Manual", "Optuna Tuning", "Automated"))
            task_type = st.radio("Select task type:", ("Regression", "Classification"))
            
            if model_flow == "Manual":
                if task_type == "Regression":
                    model_choice = st.selectbox("Select regression model:", [
                        "Linear Regression", "Lasso Regression", "Ridge Regression", "ElasticNet Regression",
                        "Decision Tree", "Random Forest", "Gradient Boosting", "AdaBoost",
                        "SVR", "XGBoost Regressor"
                    ])
                    
                    if st.button("Run Model"):
                        try:
                            if model_choice == "Linear Regression":
                                results = manual.Linear_regressor(X_train, X_test, y_train, y_test)
                            elif model_choice == "Lasso Regression":
                                results = manual.Lasso_regression(X_train, X_test, y_train, y_test)
                            elif model_choice == "Ridge Regression":
                                results = manual.Ridge_regression(X_train, X_test, y_train, y_test)
                            elif model_choice == "ElasticNet Regression":
                                results = manual.ElasticNet_regression(X_train, X_test, y_train, y_test)
                            elif model_choice == "Decision Tree":
                                results = manual.Decision_tree_regression(X_train, X_test, y_train, y_test)
                            elif model_choice == "Random Forest":
                                results = manual.Random_forest_regression(X_train, X_test, y_train, y_test)
                            elif model_choice == "Gradient Boosting":
                                results = manual.Gradient_boosting_regressor(X_train, X_test, y_train, y_test)
                            elif model_choice == "AdaBoost":
                                results = manual.AdaBoost_regressor(X_train, X_test, y_train, y_test)
                            elif model_choice == "SVR":
                                results = manual.SVR_regression(X_train, X_test, y_train, y_test)
                            elif model_choice == "XGBoost Regressor":
                                results = manual.Xgb_regressor(X_train, X_test, y_train, y_test)

                            model, predictions, r2_score, mae, mse, rmse = results

                            st.session_state.trained_model = model
                            st.session_state.model_parameters = {
                                'model_name': model_choice,
                                'model_type': 'regression',
                                'training_method': 'manual',
                                'random_state': 42,
                                'performance_metrics': {
                                    'r2_score': r2_score,
                                    'mae': mae,
                                    'mse': mse,
                                    'rmse': rmse
                                }
                            }
                            st.session_state.predictions = predictions
                            st.session_state.model_trained = True
                            
                            st.success("Model trained successfully!")
                            st.write(f"**R² Score:** {r2_score:.4f}")
                            st.write(f"**MAE:** {mae:.4f}")
                            st.write(f"**MSE:** {mse:.4f}")
                            st.write(f"**RMSE:** {rmse:.4f}")
                            
                        except Exception as e:
                            st.error(f"Error in model training: {str(e)}")
                    
                else:  
                    model_choice = st.selectbox("Select classification model:", [
                        "Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting",
                        "AdaBoost", "SVR Classifier", "KNN Classifier", "XGBoost Classifier"
                    ])
                    
                    if st.button("Train Model"):
                        try:
                            if model_choice == "Logistic Regression":
                                results = manual.Logistic_regression(X_train, X_test, y_train, y_test)
                            elif model_choice == "Decision Tree":
                                results = manual.Decision_tree_classifier(X_train, X_test, y_train, y_test)
                            elif model_choice == "Random Forest":
                                results = manual.Random_forest_classifier(X_train, X_test, y_train, y_test)
                            elif model_choice == "Gradient Boosting":
                                results = manual.Gradient_boosting_classifier(X_train, X_test, y_train, y_test)
                            elif model_choice == "AdaBoost":
                                results = manual.AdaBoost_classifier(X_train, X_test, y_train, y_test)
                            elif model_choice == "SVR Classifier":
                                results = manual.SVR_classifier(X_train, X_test, y_train, y_test)
                            elif model_choice == "KNN Classifier":
                                results = manual.KNN_classifier(X_train, X_test, y_train, y_test)
                            elif model_choice == "XGBoost Classifier":
                                results = manual.XGBoost_classifier(X_train, X_test, y_train, y_test)
                                        
                            model, predictions, accuracy, f1_score = results
                            
                            st.session_state.trained_model = model
                            st.session_state.model_parameters = {
                                'model_name': model_choice,
                                'model_type': 'classification',
                                'training_method': 'manual',
                                'performance_metrics': {
                                    'accuracy': accuracy,
                                    'f1_score': f1_score
                                }
                            }
                            st.session_state.predictions = predictions
                            st.session_state.model_trained = True
                            
                            st.success("Model trained successfully!")
                            st.write(f"Accuracy: {accuracy:.4f}")
                            st.write(f"F1 Score: {f1_score:.4f}")
                            
                        except Exception as e:
                            st.error(f"Error in model training: {str(e)}")
                            
            elif model_flow == 'Optuna Tuning':
                if task_type == "Regression":
                    model_choice = st.selectbox("Select regression model:", [
                        "Ridge Regression", "ElasticNetRegression", "Decision Tree", "Random Forest", 
                        "Gradient Boosting", "AdaBoost", "SVR", "XGBoost Regressor", 
                        "KNN Regressor", "GaussianProcessRegressor", "Ensemble Model", "PLS Regression" , "ANN_regression"])
                    
                    if st.button("Run Model"):
                        try:
                            if model_choice == "Ridge Regression":
                                results = optuna.Ridge_regression(X_train, X_test, y_train, y_test )
                            elif model_choice == "ElasticNetRegression":
                                results = optuna.ElasticNet_regression(X_train, X_test, y_train, y_test)
                            elif model_choice == "Decision Tree":
                                results = optuna.Decision_tree_regression(X_train, X_test, y_train, y_test)
                            elif model_choice == "Random Forest":
                                results = optuna.Random_forest_regression(X_train, X_test, y_train, y_test)
                            elif model_choice == "Gradient Boosting":
                                results = optuna.Gradient_boosting_regressor(X_train, X_test, y_train, y_test)
                            elif model_choice == "AdaBoost":
                                results = optuna.AdaBoost_regressor(X_train, X_test, y_train, y_test)
                            elif model_choice == "SVR":
                                results = optuna.SVR_regression(X_train, X_test, y_train, y_test)
                            elif model_choice == "XGBoost Regressor":
                                results = optuna.Xgb_regressor(X_train, X_test, y_train, y_test)
                            elif model_choice == "KNN Regressor":
                                results = optuna.KNN_regressor(X_train, X_test, y_train, y_test)
                            elif model_choice == "GaussianProcessRegressor":
                                results = optuna.GaussianProcess_regression(X_train, X_test, y_train, y_test)
                            elif model_choice == "Ensemble Model":
                                results = optuna.Ensemble_regressor(X_train, X_test, y_train, y_test) 
                            elif model_choice == "PLS Regression":
                                results = optuna.PLS_regression(X_train, X_test, y_train, y_test)
                            elif model_choice == "ANN_regression":
                                results = optuna.ANN_regression(X_train, X_test, y_train, y_test)
                            

                            

                            model, predictions, r2_score, mae, mse, rmse , study  = results
                            

                            st.session_state.trained_model = model
                            st.session_state.model_parameters = {
                                'model_name': model_choice,
                                'model_type': 'regression',
                                'training_method': 'optuna',
                                'random_state': 42,
                                'performance_metrics': {
                                    'r2_score': r2_score,
                                    'mae': mae,
                                    'mse': mse,
                                    'rmse': rmse
                                }
                            }
                            st.session_state.predictions = predictions
                            st.session_state.model_trained = True
                            
                            st.success("Model trained successfully!")
                            st.write(f"**R² Score:** {r2_score:.4f}")
                            st.write(f"**MAE:** {mae:.4f}")
                            st.write(f"**MSE:** {mse:.4f}")
                            st.write(f"**RMSE:** {rmse:.4f}")

                            ## Plots added 

                            import optuna.visualization as vis
                            st.subheader("Optimization History")
                            st.plotly_chart(vis.plot_optimization_history(study), use_container_width=True)

                            st.subheader("Hyperparameter Importance")
                            st.plotly_chart(vis.plot_param_importances(study), use_container_width=True)

                            st.subheader("Parallel Coordinate Plot")
                            st.plotly_chart(vis.plot_parallel_coordinate(study), use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Error in model training: {str(e)}")
                        
                else:  
                    model_choice = st.selectbox("Select classification model:", [
                        "Logistic Regression", "Random Forest", "XGBoost Classifier"
                    ])
                    
                    if st.button("Train Model"):
                        try:
                            if model_choice == "Logistic Regression":
                                results = optuna.Logistic_regression(X_train, X_test, y_train, y_test)
                            elif model_choice == "Random Forest":
                                results = optuna.Random_forest_classifier(X_train, X_test, y_train, y_test)
                            elif model_choice == "XGBoost Classifier":
                                results = optuna.XGBoost_classifier(X_train, X_test, y_train, y_test)
                            
                            model, predictions, accuracy, f1_score = results
                            
                            st.session_state.trained_model = model
                            st.session_state.model_parameters = {
                                'model_name': model_choice,
                                'model_type': 'classification',
                                'training_method': 'optuna',
                                'performance_metrics': {
                                    'accuracy': accuracy,
                                    'f1_score': f1_score
                                }
                            }
                            st.session_state.predictions = predictions
                            st.session_state.model_trained = True
                            
                            st.success("Model trained successfully!")
                            st.write(f"Accuracy: {accuracy:.4f}")
                            st.write(f"F1 Score: {f1_score:.4f}")
                            
                        except Exception as e:
                            st.error(f"Error in model training: {str(e)}")

            elif model_flow == "Automated":
                if st.button("Run Automated Model Selection"):
                    try:
                        if task_type == "Regression":
                            best_model = auto.run_regression(X_train, X_test, y_train, y_test)
                        else:
                            best_model = auto.run_classification(X_train, X_test, y_train, y_test)
                        
                        st.session_state.trained_model = best_model
                        st.session_state.model_parameters = {
                            'model_name': 'Automated Selection',
                            'model_type': task_type.lower(),
                            'training_method': 'automated'
                        }
                        st.session_state.model_trained = True
                        
                        st.success("Automated model selection completed!")
                        
                    except Exception as e:
                        st.error(f"Error in automated model selection: {str(e)}")

            # ── Save Model (.pkl) for Normal Data ────────────────────────────
            if st.session_state.model_trained:
                st.divider()
                st.subheader("💾 Save Trained Model")

                save_model_name = st.text_input(
                    "Model name:",
                    value=st.session_state.get('model_parameters', {}).get('model_name', 'model'),
                    key="normal_save_name",
                )

                if st.button("Save Model & Parameters (.pkl + .json)", key="normal_save_btn"):
                    try:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        user_name = st.session_state.get('user_name', 'user')
                        user_clean = re.sub(r'[^a-zA-Z0-9_]', '_', user_name)

                        save_dir = "saved_models"
                        os.makedirs(save_dir, exist_ok=True)

                        model_filename = os.path.join(save_dir, f"model_{user_clean}_{save_model_name}_{timestamp}.pkl")
                        json_filename = os.path.join(save_dir, f"params_{user_clean}_{save_model_name}_{timestamp}.json")

                        # Save model pickle
                        with open(model_filename, 'wb') as f:
                            pickle.dump(st.session_state.trained_model, f)

                        # Build params dict
                        def _convert(obj):
                            if isinstance(obj, np.ndarray):
                                return obj.tolist()
                            if isinstance(obj, (np.integer,)):
                                return int(obj)
                            if isinstance(obj, (np.floating,)):
                                return float(obj) if not (np.isnan(obj) or np.isinf(obj)) else None
                            if isinstance(obj, dict):
                                return {str(k): _convert(v) for k, v in obj.items()}
                            if isinstance(obj, (list, tuple)):
                                return [_convert(v) for v in obj]
                            return obj

                        params_to_save = {
                            "user_info": {
                                "user_name": user_name,
                                "creation_date": datetime.now().isoformat(),
                            },
                            "model_info": {
                                "model_name": save_model_name,
                                "model_filename": os.path.basename(model_filename),
                                "timestamp": timestamp,
                            },
                            "model_parameters": _convert(st.session_state.get('model_parameters', {})),
                            "data_info": {
                                "n_features": int(X_train.shape[1]),
                                "n_train_samples": int(X_train.shape[0]),
                                "n_test_samples": int(X_test.shape[0]),
                            },
                        }

                        with open(json_filename, 'w') as f:
                            json.dump(params_to_save, f, indent=4, default=str)

                        st.success(f"✅ Saved!  Model: `{model_filename}`  |  Params: `{json_filename}`")

                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            with open(model_filename, 'rb') as f:
                                st.download_button(
                                    "⬇️ Download Model (.pkl)",
                                    f.read(),
                                    file_name=os.path.basename(model_filename),
                                    mime="application/octet-stream",
                                    key="dl_model_normal",
                                )
                        with col_dl2:
                            with open(json_filename, 'r') as f:
                                st.download_button(
                                    "⬇️ Download Parameters (.json)",
                                    f.read(),
                                    file_name=os.path.basename(json_filename),
                                    mime="application/json",
                                    key="dl_params_normal",
                                )
                    except Exception as e:
                        st.error(f"Error saving model: {e}")
                        import traceback
                        st.error(traceback.format_exc())

        else :
        
            if hasattr(y_train, 'values'):
                y_train_flat = y_train.values.ravel() if len(y_train.shape) > 1 else y_train.values
            else:
                y_train_flat = y_train.ravel() if len(y_train.shape) > 1 else y_train
            
            if hasattr(y_test, 'values'):
                y_test_flat = y_test.values.ravel() if len(y_test.shape) > 1 else y_test.values
            else:
                y_test_flat = y_test.ravel() if len(y_test.shape) > 1 else y_test
            
            st.write(f"Training data ready: Train={X_train.shape}, Test={X_test.shape}")
            st.write(f"Target shapes: y_train={y_train_flat.shape}, y_test={y_test_flat.shape}")
            
            st.subheader("Zero-Inflation Analysis")
            zero_count_train = np.sum(y_train_flat == 0)
            nonzero_count_train = np.sum(y_train_flat != 0)
            zero_percentage = (zero_count_train / len(y_train_flat)) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Zero values", zero_count_train)
            with col2:
                st.metric("Non-zero values", nonzero_count_train)
            with col3:
                st.metric("Zero percentage", f"{zero_percentage:.1f}%")
            
            if zero_percentage < 10:
                st.warning("Low zero-inflation detected. Consider regular regression instead.")
            elif zero_percentage > 80:
                st.warning("Very high zero-inflation. Check data quality.")
            else:
                st.success("Good candidate for zero-inflated modeling")
            
            if ('augmented_X_train' in st.session_state and 
                st.session_state.augmented_X_train is not None and 
                st.session_state.augmented_X_train.size > 0 and 
                X_train.shape[0] == st.session_state.augmented_X_train.shape[0]):
                
                original_size = st.session_state.X_train.shape[0] if hasattr(st.session_state, 'X_train') else 'Unknown'
                augmented_size = X_train.shape[0]
                st.success(f"Augmented Training Data Active: {original_size} -> {augmented_size} samples")
                
                if hasattr(st.session_state, 'augmentation_history') and st.session_state.augmentation_history:
                    with st.expander("View Augmentation History"):
                        for i, aug in enumerate(st.session_state.augmentation_history):
                            st.write(f"{i+1}. {aug}")
            
            if X_train.shape[0] != y_train_flat.shape[0] or X_test.shape[0] != y_test_flat.shape[0]:
                st.error(f"Data inconsistency detected")
                st.error(f"Training: X={X_train.shape[0]} samples, y={y_train_flat.shape[0]} targets")
                st.error(f"Testing: X={X_test.shape[0]} samples, y={y_test_flat.shape[0]} targets")
                return
            
            from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
            
            y_binary_train = (y_train_flat != 0).astype(int)
            y_binary_test = (y_test_flat != 0).astype(int)
            
            nonzero_mask_train = y_train_flat != 0
            nonzero_mask_test = y_test_flat != 0
            
            X_reg_train = X_train[nonzero_mask_train]
            y_reg_train = y_train_flat[nonzero_mask_train]
            X_reg_test = X_test[nonzero_mask_test]
            y_reg_test = y_test_flat[nonzero_mask_test]
            
            st.write(f"Binary classification data: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
            st.write(f"Regression data (non-zero only): Train={X_reg_train.shape[0]}, Test={X_reg_test.shape[0]}")
            
            model_flow = st.radio("Select model run type", ("Manual", "Optuna Tuning", "Automated"))
            
            def create_combined_predictions_and_metrics(clf_model, reg_model, X_train, X_test, y_train_flat, y_test_flat, y_binary_train, y_binary_test, X_reg_train, y_reg_train, X_reg_test, y_reg_test):
                clf_train_pred = clf_model.predict(X_train)
                clf_test_pred = clf_model.predict(X_test)
                
                reg_train_pred = reg_model.predict(X_reg_train)
                reg_test_pred = reg_model.predict(X_reg_test)
                
                combined_train_pred = np.zeros_like(y_train_flat, dtype=float)
                combined_test_pred = np.zeros_like(y_test_flat, dtype=float)
                
                train_nonzero_mask = clf_train_pred == 1
                test_nonzero_mask = clf_test_pred == 1
                
                if np.any(train_nonzero_mask):
                    X_train_nonzero = X_train[train_nonzero_mask]
                    if len(X_train_nonzero) > 0:
                        train_reg_pred = reg_model.predict(X_train_nonzero)
                        combined_train_pred[train_nonzero_mask] = train_reg_pred
                
                if np.any(test_nonzero_mask):
                    X_test_nonzero = X_test[test_nonzero_mask]
                    if len(X_test_nonzero) > 0:
                        test_reg_pred = reg_model.predict(X_test_nonzero)
                        combined_test_pred[test_nonzero_mask] = test_reg_pred
                
                clf_train_acc = accuracy_score(y_binary_train, clf_train_pred)
                clf_test_acc = accuracy_score(y_binary_test, clf_test_pred)
                clf_train_f1 = f1_score(y_binary_train, clf_train_pred)
                clf_test_f1 = f1_score(y_binary_test, clf_test_pred)
                
                reg_train_r2 = r2_score(y_reg_train, reg_train_pred)
                reg_test_r2 = r2_score(y_reg_test, reg_test_pred)
                reg_train_mae = mean_absolute_error(y_reg_train, reg_train_pred)
                reg_test_mae = mean_absolute_error(y_reg_test, reg_test_pred)
                reg_train_rmse = np.sqrt(mean_squared_error(y_reg_train, reg_train_pred))
                reg_test_rmse = np.sqrt(mean_squared_error(y_reg_test, reg_test_pred))
                
                overall_train_r2 = r2_score(y_train_flat, combined_train_pred)
                overall_test_r2 = r2_score(y_test_flat, combined_test_pred)
                overall_train_mae = mean_absolute_error(y_train_flat, combined_train_pred)
                overall_test_mae = mean_absolute_error(y_test_flat, combined_test_pred)
                overall_train_rmse = np.sqrt(mean_squared_error(y_train_flat, combined_train_pred))
                overall_test_rmse = np.sqrt(mean_squared_error(y_test_flat, combined_test_pred))
                
                return {
                    'train_metrics': {
                        'classification': {'accuracy': clf_train_acc, 'f1_score': clf_train_f1},
                        'regression': {'r2_score': reg_train_r2, 'mae': reg_train_mae, 'rmse': reg_train_rmse},
                        'overall': {'r2_score': overall_train_r2, 'mae': overall_train_mae, 'rmse': overall_train_rmse}
                    },
                    'test_metrics': {
                        'classification': {'accuracy': clf_test_acc, 'f1_score': clf_test_f1},
                        'regression': {'r2_score': reg_test_r2, 'mae': reg_test_mae, 'rmse': reg_test_rmse},
                        'overall': {'r2_score': overall_test_r2, 'mae': overall_test_mae, 'rmse': overall_test_rmse}
                    },
                    'predictions': {
                        'binary_test': clf_test_pred,
                        'regression_test': reg_test_pred,
                        'combined_test': combined_test_pred,
                        'combined_train': combined_train_pred
                    }
                }
            
            def plot_train_test_comparison(train_metrics, test_metrics):
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
                
                categories = ['Classification\nAccuracy', 'Classification\nF1 Score']
                train_clf = [train_metrics['classification']['accuracy'], train_metrics['classification']['f1_score']]
                test_clf = [test_metrics['classification']['accuracy'], test_metrics['classification']['f1_score']]
                
                x = np.arange(len(categories))
                width = 0.35
                
                ax1.bar(x - width/2, train_clf, width, label='Train', color='skyblue')
                ax1.bar(x + width/2, test_clf, width, label='Test', color='lightcoral')
                ax1.set_ylabel('Score')
                ax1.set_title('Classification Performance')
                ax1.set_xticks(x)
                ax1.set_xticklabels(categories)
                ax1.legend()
                ax1.set_ylim(0, 1)
                
                reg_categories = ['R² Score', 'MAE', 'RMSE']
                train_reg = [train_metrics['regression']['r2_score'], train_metrics['regression']['mae'], train_metrics['regression']['rmse']]
                test_reg = [test_metrics['regression']['r2_score'], test_metrics['regression']['mae'], test_metrics['regression']['rmse']]
                
                x2 = np.arange(len(reg_categories))
                ax2.bar(x2 - width/2, train_reg, width, label='Train', color='skyblue')
                ax2.bar(x2 + width/2, test_reg, width, label='Test', color='lightcoral')
                ax2.set_ylabel('Score')
                ax2.set_title('Regression Performance (Non-Zero Values)')
                ax2.set_xticks(x2)
                ax2.set_xticklabels(reg_categories)
                ax2.legend()
                
                overall_categories = ['R² Score', 'MAE', 'RMSE']
                train_overall = [train_metrics['overall']['r2_score'], train_metrics['overall']['mae'], train_metrics['overall']['rmse']]
                test_overall = [test_metrics['overall']['r2_score'], test_metrics['overall']['mae'], test_metrics['overall']['rmse']]
                
                x3 = np.arange(len(overall_categories))
                ax3.bar(x3 - width/2, train_overall, width, label='Train', color='skyblue')
                ax3.bar(x3 + width/2, test_overall, width, label='Test', color='lightcoral')
                ax3.set_ylabel('Score')
                ax3.set_title('Overall Combined Performance')
                ax3.set_xticks(x3)
                ax3.set_xticklabels(overall_categories)
                ax3.legend()
                
                r2_comparison = [train_metrics['overall']['r2_score'], test_metrics['overall']['r2_score']]
                ax4.bar(['Train R²', 'Test R²'], r2_comparison, color=['skyblue', 'lightcoral'])
                ax4.set_ylabel('R² Score')
                ax4.set_title('Train vs Test R² Comparison')
                ax4.set_ylim(0, max(1, max(r2_comparison) * 1.1))
                
                for i, v in enumerate(r2_comparison):
                    ax4.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            
            manual = Models()
            optuna = optuna_Model()
            auto = AutoModelSelector()
            
            if model_flow == "Manual":
                st.subheader("Model Selection")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Step 1: Classification Model (Zero vs Non-Zero)")
                    classification_model = st.selectbox("Select classification model:", [
                        "Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting",
                        "AdaBoost", "SVM Classifier", "XGBoost Classifier"
                    ], key="clf_model")
                
                with col2:
                    st.write("Step 2: Regression Model (Non-Zero Values)")
                    regression_model = st.selectbox("Select regression model:", [
                        "Linear Regression", "Lasso Regression", "Ridge Regression", "ElasticNet Regression",
                        "Decision Tree", "Random Forest", "Gradient Boosting", "AdaBoost",
                        "SVR", "XGBoost Regressor"
                    ], key="reg_model")
                
                if st.button("Train Two-Part Model"):
                    try:
                        st.write("Training Classification Model...")
                        if classification_model == "Logistic Regression":
                            clf_results = manual.Logistic_regression(X_train, X_test, y_binary_train, y_binary_test)
                        elif classification_model == "Decision Tree":
                            clf_results = manual.Decision_tree_classifier(X_train, X_test, y_binary_train, y_binary_test)
                        elif classification_model == "Random Forest":
                            clf_results = manual.Random_forest_classifier(X_train, X_test, y_binary_train, y_binary_test)
                        elif classification_model == "Gradient Boosting":
                            clf_results = manual.Gradient_boosting_classifier(X_train, X_test, y_binary_train, y_binary_test)
                        elif classification_model == "AdaBoost":
                            clf_results = manual.AdaBoost_classifier(X_train, X_test, y_binary_train, y_binary_test)
                        elif classification_model == "SVM Classifier":
                            clf_results = manual.SVR_classifier(X_train, X_test, y_binary_train, y_binary_test)
                        elif classification_model == "XGBoost Classifier":
                            clf_results = manual.XGBoost_classifier(X_train, X_test, y_binary_train, y_binary_test)
                        
                        clf_model, clf_predictions, clf_accuracy, clf_f1 = clf_results
                        st.success(f"Classification Model trained. Accuracy: {clf_accuracy:.4f}, F1: {clf_f1:.4f}")
                        
                        st.write("Training Regression Model...")
                        if len(X_reg_train) == 0:
                            st.error("No non-zero training data available for regression")
                            return
                        
                        if regression_model == "Linear Regression":
                            reg_results = manual.Linear_regressor(X_reg_train, X_reg_test, y_reg_train, y_reg_test)
                        elif regression_model == "Lasso Regression":
                            reg_results = manual.Lasso_regression(X_reg_train, X_reg_test, y_reg_train, y_reg_test)
                        elif regression_model == "Ridge Regression":
                            reg_results = manual.Ridge_regression(X_reg_train, X_reg_test, y_reg_train, y_reg_test)
                        elif regression_model == "ElasticNet Regression":
                            reg_results = manual.ElasticNet_regression(X_reg_train, X_reg_test, y_reg_train, y_reg_test)
                        elif regression_model == "Decision Tree":
                            reg_results = manual.Decision_tree_regression(X_reg_train, X_reg_test, y_reg_train, y_reg_test)
                        elif regression_model == "Random Forest":
                            reg_results = manual.Random_forest_regression(X_reg_train, X_reg_test, y_reg_train, y_reg_test)
                        elif regression_model == "Gradient Boosting":
                            reg_results = manual.Gradient_boosting_regressor(X_reg_train, X_reg_test, y_reg_train, y_reg_test)
                        elif regression_model == "AdaBoost":
                            reg_results = manual.AdaBoost_regressor(X_reg_train, X_reg_test, y_reg_train, y_reg_test)
                        elif regression_model == "SVR":
                            reg_results = manual.SVR_regression(X_reg_train, X_reg_test, y_reg_train, y_reg_test)
                        elif regression_model == "XGBoost Regressor":
                            reg_results = manual.Xgb_regressor(X_reg_train, X_reg_test, y_reg_train, y_reg_test)
                        
                        reg_model, reg_predictions, reg_r2, reg_mae, reg_mse, reg_rmse = reg_results
                        st.success(f"Regression Model trained. R²: {reg_r2:.4f}, RMSE: {reg_rmse:.4f}")
                        
                        st.write("Combining predictions...")
                        metrics_data = create_combined_predictions_and_metrics(clf_model, reg_model, X_train, X_test, y_train_flat, y_test_flat, y_binary_train, y_binary_test, X_reg_train, y_reg_train, X_reg_test, y_reg_test)
                        
                        st.session_state.two_part_models = {
                            'classification_model': clf_model,
                            'regression_model': reg_model,
                            'classification_name': classification_model,
                            'regression_name': regression_model,
                            'training_method': 'manual',
                            'performance_metrics': metrics_data['test_metrics'],
                            'train_metrics': metrics_data['train_metrics'],
                            'predictions': metrics_data['predictions']
                        }
                        
                        st.session_state.trained_model = st.session_state.two_part_models
                        st.session_state.model_parameters = {
                            'model_name': 'Zero-Inflated Model',
                            'model_type': 'zero_inflated',
                            'training_method': 'manual',
                            'classification_model': classification_model,
                            'regression_model': regression_model,
                            'performance_metrics': metrics_data['test_metrics']
                        }
                        st.session_state.predictions = metrics_data['predictions']['combined_test']
                        st.session_state.model_trained = True
                        
                        st.subheader("Two-Part Model Results")
                        plot_train_test_comparison(metrics_data['train_metrics'], metrics_data['test_metrics'])
                        
                    except Exception as e:
                        st.error(f"Error in two-part model training: {str(e)}")
                        import traceback
                        st.error(f"Traceback: {traceback.format_exc()}")
            
            elif model_flow == 'Optuna Tuning':
                st.subheader("Optuna-Optimized Two-Part Model")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Step 1: Classification Model (Zero vs Non-Zero)")
                    classification_model = st.selectbox("Select classification model:", [
                        "Logistic Regression", "Random Forest", "XGBoost Classifier"
                    ], key="optuna_clf_model")
                
                with col2:
                    st.write("Step 2: Regression Model (Non-Zero Values)")
                    regression_model = st.selectbox("Select regression model:", [
                        "Ridge Regression", "ElasticNet Regression", "Decision Tree", "Random Forest", 
                        "Gradient Boosting", "AdaBoost", "SVR", "XGBoost Regressor"
                    ], key="optuna_reg_model")
                
                if st.button("Train Optuna Two-Part Model"):
                    try:
                        st.write("Training Classification Model with Optuna...")
                        if classification_model == "Logistic Regression":
                            clf_results = optuna.Logistic_regression(X_train, X_test, y_binary_train, y_binary_test)
                        elif classification_model == "Random Forest":
                            clf_results = optuna.Random_forest_classifier(X_train, X_test, y_binary_train, y_binary_test)
                        elif classification_model == "XGBoost Classifier":
                            clf_results = optuna.XGBoost_classifier(X_train, X_test, y_binary_train, y_binary_test)
                        
                        clf_model, clf_predictions, clf_accuracy, clf_f1 = clf_results
                        st.success(f"Classification Model trained. Accuracy: {clf_accuracy:.4f}, F1: {clf_f1:.4f}")
                        
                        st.write("Training Regression Model with Optuna...")
                        if len(X_reg_train) == 0:
                            st.error("No non-zero training data available for regression")
                            return
                        
                        if regression_model == "Ridge Regression":
                            reg_results = optuna.Ridge_regression(X_reg_train, X_reg_test, y_reg_train, y_reg_test)
                        elif regression_model == "ElasticNet Regression":
                            reg_results = optuna.ElasticNet_regression(X_reg_train, X_reg_test, y_reg_train, y_reg_test)
                        elif regression_model == "Decision Tree":
                            reg_results = optuna.Decision_tree_regression(X_reg_train, X_reg_test, y_reg_train, y_reg_test)
                        elif regression_model == "Random Forest":
                            reg_results = optuna.Random_forest_regression(X_reg_train, X_reg_test, y_reg_train, y_reg_test)
                        elif regression_model == "Gradient Boosting":
                            reg_results = optuna.Gradient_boosting_regressor(X_reg_train, X_reg_test, y_reg_train, y_reg_test)
                        elif regression_model == "AdaBoost":
                            reg_results = optuna.AdaBoost_regressor(X_reg_train, X_reg_test, y_reg_train, y_reg_test)
                        elif regression_model == "SVR":
                            reg_results = optuna.SVR_regression(X_reg_train, X_reg_test, y_reg_train, y_reg_test)
                        elif regression_model == "XGBoost Regressor":
                            reg_results = optuna.Xgb_regressor(X_reg_train, X_reg_test, y_reg_train, y_reg_test)
                        
                        reg_model, reg_predictions, reg_r2, reg_mae, reg_mse, reg_rmse, study = reg_results
                        
                        st.success(f"Regression Model trained. R²: {reg_r2:.4f}, RMSE: {reg_rmse:.4f}")
                        
                        st.write("Combining predictions...")
                        metrics_data = create_combined_predictions_and_metrics(clf_model, reg_model, X_train, X_test, y_train_flat, y_test_flat, y_binary_train, y_binary_test, X_reg_train, y_reg_train, X_reg_test, y_reg_test)
                        
                        st.session_state.two_part_models = {
                            'classification_model': clf_model,
                            'regression_model': reg_model,
                            'classification_name': classification_model,
                            'regression_name': regression_model,
                            'training_method': 'optuna',
                            'performance_metrics': metrics_data['test_metrics'],
                            'train_metrics': metrics_data['train_metrics'],
                            'predictions': metrics_data['predictions']
                        }
                        
                        st.session_state.trained_model = st.session_state.two_part_models
                        st.session_state.model_parameters = {
                            'model_name': 'Zero-Inflated Model (Optuna)',
                            'model_type': 'zero_inflated',
                            'training_method': 'optuna',
                            'classification_model': classification_model,
                            'regression_model': regression_model,
                            'performance_metrics': metrics_data['test_metrics']
                        }
                        st.session_state.predictions = metrics_data['predictions']['combined_test']
                        st.session_state.model_trained = True
                        
                        st.subheader("Optuna Two-Part Model Results")
                        plot_train_test_comparison(metrics_data['train_metrics'], metrics_data['test_metrics'])
                        
                    except Exception as e:
                        st.error(f"Error in Optuna two-part model training: {str(e)}")
                        import traceback
                        st.error(f"Traceback: {traceback.format_exc()}")
            
            elif model_flow == "Automated":
                st.subheader("Automated Two-Part Model Selection")
                
                if st.button("Run Automated Two-Part Model Selection"):
                    try:
                        st.write("Running automated classification model selection...")
                        best_clf_model = auto.run_classification(X_train, X_test, y_binary_train, y_binary_test)
                        
                        st.write("Running automated regression model selection...")
                        if len(X_reg_train) == 0:
                            st.error("No non-zero training data available for regression")
                            return
                        
                        best_reg_model = auto.run_regression(X_reg_train, X_reg_test, y_reg_train, y_reg_test)
                        
                        st.write("Combining automated models...")
                        metrics_data = create_combined_predictions_and_metrics(best_clf_model, best_reg_model, X_train, X_test, y_train_flat, y_test_flat, y_binary_train, y_binary_test, X_reg_train, y_reg_train, X_reg_test, y_reg_test)
                        
                        st.session_state.two_part_models = {
                            'classification_model': best_clf_model,
                            'regression_model': best_reg_model,
                            'classification_name': 'Automated Selection',
                            'regression_name': 'Automated Selection',
                            'training_method': 'automated',
                            'performance_metrics': metrics_data['test_metrics'],
                            'train_metrics': metrics_data['train_metrics'],
                            'predictions': metrics_data['predictions']
                        }
                        
                        st.session_state.trained_model = st.session_state.two_part_models
                        st.session_state.model_parameters = {
                            'model_name': 'Zero-Inflated Model (Automated)',
                            'model_type': 'zero_inflated',
                            'training_method': 'automated',
                            'classification_model': 'Automated Selection',
                            'regression_model': 'Automated Selection',
                            'performance_metrics': metrics_data['test_metrics']
                        }
                        st.session_state.predictions = metrics_data['predictions']['combined_test']
                        st.session_state.model_trained = True
                        
                        st.success("Automated two-part model selection completed")
                        
                        st.subheader("Automated Two-Part Model Results")
                        plot_train_test_comparison(metrics_data['train_metrics'], metrics_data['test_metrics'])
                        
                    except Exception as e:
                        st.error(f"Error in automated two-part model selection: {str(e)}")
                        import traceback
                        st.error(f"Traceback: {traceback.format_exc()}")
            
            if st.session_state.model_trained:
                st.subheader("Save Model")
                
                model_name = st.text_input("Enter model name for saving", value="zero_inflated_model")
                
                if st.button("Save Model and Parameters"):
                    try:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        save_dir = "saved_models"
                        os.makedirs(save_dir, exist_ok=True)
                        
                        model_filename = f"{save_dir}/{model_name}_{timestamp}.pkl"
                        params_filename = f"{save_dir}/{model_name}_{timestamp}_params.json"
                        
                        model_package = {
                            'classification_model': st.session_state.two_part_models['classification_model'],
                            'regression_model': st.session_state.two_part_models['regression_model'],
                            'model_metadata': {
                                'classification_name': st.session_state.two_part_models['classification_name'],
                                'regression_name': st.session_state.two_part_models['regression_name'],
                                'training_method': st.session_state.two_part_models['training_method'],
                                'timestamp': timestamp
                            }
                        }
                        
                        with open(model_filename, 'wb') as f:
                            pickle.dump(model_package, f)
                        
                        params_data = {
                            'model_name': model_name,
                            'model_type': 'zero_inflated',
                            'timestamp': timestamp,
                            'classification_model': st.session_state.two_part_models['classification_name'],
                            'regression_model': st.session_state.two_part_models['regression_name'],
                            'training_method': st.session_state.two_part_models['training_method'],
                            'performance_metrics': st.session_state.two_part_models['performance_metrics'],
                            'train_metrics': st.session_state.two_part_models['train_metrics'],
                            'data_info': {
                                'n_features': X_train.shape[1],
                                'n_train_samples': X_train.shape[0],
                                'n_test_samples': X_test.shape[0],
                                'zero_percentage': zero_percentage
                            }
                        }
                        
                        with open(params_filename, 'w') as f:
                            json.dump(params_data, f, indent=4, default=str)
                        
                        st.success(f"Model saved successfully")
                        st.info(f"Model file: {model_filename}")
                        st.info(f"Parameters file: {params_filename}")
                        
                    except Exception as e:
                        st.error(f"Error saving model: {str(e)}")
                        import traceback
                        st.error(f"Traceback: {traceback.format_exc()}")
                
                if st.button("Proceed to New Model Training"):
                    st.session_state.step = 1
                    st.session_state.model_trained = False
                    st.session_state.data_loaded = False
                    st.session_state.targets_set = False
                    st.session_state.data_split = False
                    st.rerun()


app()

from chatbot import render_chatbot
render_chatbot("03_Model_Training")

