import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from data_augmentation import DataAugmentor
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
from spectra_specific.RamanSpectra1 import RamanPreprocessingOptimizer
from spectra_specific.FTIRSpectra import FTIRPreprocessingOptimizer
from FFT import FFTProcessor
import copy 
from copy import deepcopy
from mpls import MultiWayPLS


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {str(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, set):
        return [convert_numpy_types(v) for v in sorted(obj, key=str)]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    else:
        return obj


    
def reset_to_step(target_step):
    """Reset session state to a specific step"""
    steps_to_reset = {
        1: ['data_loaded', 'current_data', 'original_data'],
        2: ['targets_set', 'target_columns', 'X_full', 'y_full', 'current_X'],
        3: ['outliers_processed'],
        4: ['data_split', 'X_train', 'X_test', 'y_train', 'y_test'],
        5: ['wavelet_done', 'fitted_denoiser'],
        6: ['preprocessing_done', 'spectral_data_train', 'spectral_data_test', 'preprocessing_history'],
        7: ['dimensionality_done', 'dim_reducer', 'dimensionality_history'],
        8: ['augmentation_done', 'augmented_X_train', 'augmented_y_train', 'augmentation_history'],
        9: [],
        10:['model_trained', 'trained_model', 'model_parameters', 'predictions'],
        11:['model_evaluated'],
        12:['model_saved']
    }
    
    # Reset all steps after the target step
    for step in range(target_step + 1, 14):
        if step in steps_to_reset:
            for key in steps_to_reset[step]:
                if key in st.session_state:
                    del st.session_state[key]
    
    # Clear data history for steps after target
    if 'data_history' in st.session_state:
        history_keys = list(st.session_state.data_history.keys())
        for key in history_keys:
            # This is a simplified approach - you might want to be more specific
            if target_step <= 4:  # Reset everything if going back to early steps
                if key in st.session_state.data_history:
                    del st.session_state.data_history[key]
    
    st.session_state.step = target_step

def main():
    st.title("Multi-Model ML Pipeline")
    
    rd = ReadingData()
    manual = Models()
    optuna = optuna_Model()
    auto = AutoModelSelector()
    
    # Initialize all session state variables
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'targets_set' not in st.session_state: 
        st.session_state.targets_set = False
    if 'wavelet_done' not in st.session_state:
        st.session_state.wavelet_done = False
    if 'outliers_processed' not in st.session_state:
        st.session_state.outliers_processed = False
    if 'preprocessing_done' not in st.session_state:
        st.session_state.preprocessing_done = False
    if 'dimensionality_done' not in st.session_state:
        st.session_state.dimensionality_done = False
    if 'data_split' not in st.session_state:
        st.session_state.data_split = False
    if 'augmentation_done' not in st.session_state:
        st.session_state.augmentation_done = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'model_evaluated' not in st.session_state:
        st.session_state.model_evaluated = False
    if 'model_saved' not in st.session_state:
        st.session_state.model_saved = False
    if 'user_name' not in st.session_state:
        st.session_state.user_name = ""
    if 'direct_prediction_mode' not in st.session_state:
        st.session_state.direct_prediction_mode = False
    if 'skipped_steps' not in st.session_state:
        st.session_state.skipped_steps = set()
    if 'preprocessing_parameters' not in st.session_state:
        st.session_state.preprocessing_parameters = {}
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
    if 'model_parameters' not in st.session_state:
        st.session_state.model_parameters = {}

    if not st.session_state.user_name:
        st.header("Welcome to Multi-Model ML Pipeline")
        st.subheader("Please enter your name to begin")
        user_name = st.text_input("Enter Your Name:", placeholder="e.g., Happy Birthday to you ")
        if user_name:
            st.session_state.user_name = user_name
            st.rerun()
        else:
            st.info("Please enter your name to proceed.")
            return

    progress_steps = ["Load Raw Data", "Target Selection" ,  "Outlier Removal", 
                     "Train-Test Split", "Wavelet Denoising", "Preprocessing", 
                     "Dimensionality Reduction", "Data Augmentation", "Mpls","Model Training", 
                     "Evaluation", "Model Saving", "Preprocessing New Data", "Prediction"]
    
    st.sidebar.markdown("---")
    st.sidebar.title("Quick Start Options")
    
    if st.sidebar.button("Go Directly to Preprocessing", help="Load saved model and preprocess new data"):
        st.session_state.direct_prediction_mode = True
        st.session_state.step = 14  
        st.rerun()
    
    if st.sidebar.button("Go Directly to Prediction", help="Jump directly to prediction with preprocessed data"):
        st.session_state.direct_prediction_mode = True
        st.session_state.step = 14  
        st.rerun()
    
    if st.sidebar.button("Full Pipeline Mode", help="Start from data upload"):
        st.session_state.direct_prediction_mode = False
        st.session_state.step = 1
        st.rerun()
    
    st.sidebar.title("Pipeline Progress")
    for i, step in enumerate(progress_steps, 1):
        if i in st.session_state.skipped_steps:
            st.sidebar.warning(f"Step {i}: {step} (Skipped)")
        elif st.session_state.step > i:
            st.sidebar.success(f"Step {i}: {step}")
        elif st.session_state.step == i:
            st.sidebar.info(f"Step {i}: {step}")
        else:
            st.sidebar.write(f"Step {i}: {step}")

    # Initialize data history to track data at each step
    
    if 'data_history' not in st.session_state:
        st.session_state.data_history = {}

    # Helper functions
    def save_step_data(step_name, X_train, X_test, y_train, y_test):
        """Save train/test data at a specific step"""
        st.session_state.data_history[step_name] = {
            'X_train': X_train.copy() if X_train is not None else None,
            'X_test': X_test.copy() if X_test is not None else None,
            'y_train': y_train.copy() if y_train is not None else None,
            'y_test': y_test.copy() if y_test is not None else None
        }
    
    def get_step_data(step_name):
        """Get train/test data from a specific step"""
        if step_name in st.session_state.data_history:
            data = st.session_state.data_history[step_name]
            return data['X_train'], data['X_test'], data['y_train'], data['y_test']
        return None, None, None, None
    
    def get_current_data():
        """Get the most recent version of processed train/test data with better fallback"""
        for step_name in ['augmentation', 'dimensionality', 'preprocessing', 'wavelet', 'split']:
            if step_name in st.session_state.data_history:
                X_train, X_test, y_train, y_test = get_step_data(step_name)
                if X_train is not None:
                    return X_train, X_test, y_train, y_test
        
        # Final fallback to session state
        if all(hasattr(st.session_state, attr) for attr in ['X_train', 'X_test', 'y_train', 'y_test']):
            return st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test
        
        return None, None, None, None



    def save_model_and_parameters(model, model_name, model_params, preprocessing_params):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        user_name_clean = re.sub(r'[^a-zA-Z0-9_]', '_', st.session_state.user_name) if st.session_state.user_name else "unknown_user"
        
        model_filename = f"model_{user_name_clean}_{model_name}_{timestamp}.pkl"
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        
        # Save fitted preprocessing objects for reuse during prediction
        fitted_objects = {}
        if hasattr(st.session_state, 'fitted_denoiser') and st.session_state.fitted_denoiser is not None:
            try:
                pickle.dumps(st.session_state.fitted_denoiser)  # test picklability
                fitted_objects['wavelet_denoiser'] = st.session_state.fitted_denoiser
            except Exception:
                pass  # skip unpicklable objects
        if hasattr(st.session_state, 'dim_reducer') and st.session_state.dim_reducer is not None:
            dim_reducer = st.session_state.dim_reducer
            try:
                if dim_reducer.scaler is not None:
                    pickle.dumps(dim_reducer.scaler)
                    fitted_objects['dim_reducer_scaler'] = dim_reducer.scaler
            except Exception:
                pass
            try:
                if dim_reducer.reducer is not None:
                    pickle.dumps(dim_reducer.reducer)
                    fitted_objects['dim_reducer_reducer'] = dim_reducer.reducer
            except Exception:
                pass
        if hasattr(st.session_state, 'auto_optimizer') and st.session_state.auto_optimizer is not None:
            try:
                pickle.dumps(st.session_state.auto_optimizer)  # test picklability
                fitted_objects['auto_optimizer'] = st.session_state.auto_optimizer
            except Exception:
                pass  # NIRPreprocessingOptimizer etc. may not be picklable
        
        fitted_objects_filename = None
        if fitted_objects:
            fitted_objects_filename = f"fitted_objects_{user_name_clean}_{model_name}_{timestamp}.pkl"
            try:
                with open(fitted_objects_filename, 'wb') as f:
                    pickle.dump(fitted_objects, f)
            except Exception:
                fitted_objects_filename = None  # failed to save, skip
        
        model_params_clean = convert_numpy_types(model_params)
        preprocessing_params_clean = convert_numpy_types(preprocessing_params)
        
        training_target_stats = {}
        if hasattr(st.session_state, 'y_train') and st.session_state.y_train is not None:
            y_train_flat = st.session_state.y_train.values.ravel() if hasattr(st.session_state.y_train, 'values') else st.session_state.y_train.ravel()
            training_target_stats = {
                'mean': float(np.mean(y_train_flat)),
                'median': float(np.median(y_train_flat)),
                'std': float(np.std(y_train_flat)),
                'min': float(np.min(y_train_flat)),
                'max': float(np.max(y_train_flat))
            }
        
        feature_columns = []
        if hasattr(st.session_state, 'X_train') and st.session_state.X_train is not None:
            if hasattr(st.session_state.X_train, 'columns'):
                feature_columns = [str(col) for col in st.session_state.X_train.columns]
        
        target_columns = []
        if 'target_columns' in st.session_state:
            target_columns = [str(col) for col in st.session_state.target_columns]
        
        training_shape = []
        if hasattr(st.session_state, 'X_train') and st.session_state.X_train is not None:
            training_shape = list(st.session_state.X_train.shape)
        
        test_shape = []
        if hasattr(st.session_state, 'X_test') and st.session_state.X_test is not None:
            test_shape = list(st.session_state.X_test.shape)
        
        parameters = {
            "user_info": {
                "user_name": st.session_state.user_name,
                "creation_date": datetime.now().isoformat(),
                "user_id": user_name_clean
            },
            "model_info": {
                "model_name": str(model_name),
                "timestamp": timestamp,
                "model_filename": model_filename,
                "fitted_objects_filename": fitted_objects_filename
            },
            "model_parameters": model_params_clean,
            "preprocessing_parameters": preprocessing_params_clean,
            "training_target_stats": training_target_stats,
            "data_info": {
                "feature_columns": feature_columns,
                "target_columns": target_columns,
                "training_shape": training_shape,
                "test_shape": test_shape
            }
        }
        
        parameters = convert_numpy_types(parameters)
        
        json_filename = f"parameters_{user_name_clean}_{model_name}_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(parameters, f, indent=4, ensure_ascii=False, default=str)
        
        return model_filename, json_filename



    # Step 1: Load Raw Data
    if st.session_state.step == 1:
        st.header("Step 1: Load Raw Data")
        
        st.subheader("User Information")
        st.success(f"Welcome, {st.session_state.user_name}!")
        
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
                
                st.success("Data loaded successfully!")
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
                # Clear all downstream data
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
                if st.button("Proceed to Outlier Removal"):
                    st.session_state.step = 3
                    st.rerun()
            with col2:
                if st.button("Skip to Train-Test Split"):
                    st.session_state.skipped_steps.add(3) 
                    st.session_state.step = 4
                    st.rerun()

    elif st.session_state.step == 3:
        st.header("Step 3: Outlier Removal")
        
        if not st.session_state.targets_set:
            st.error("Please complete target selection first")
            if st.button("Go back to Target Selection"):
                st.session_state.step = 2
                st.rerun()
            return
        
        X = st.session_state.current_X
        y = st.session_state.y_full
        x_axis = st.session_state.x_axis if 'x_axis' in st.session_state else np.arange(X.shape[1])
        
        st.subheader("Outlier Detection and Removal")
        apply_outlier_removal = st.checkbox("Apply outlier removal based on standard deviation?")
        
        if apply_outlier_removal:
            threshold = st.slider("Outlier Detection Threshold (σ)", 1.0, 5.0, 3.0, 0.1)
            
            if st.button("Remove Outliers"):
                try:
                    remover = OutlierRemover(threshold=threshold)
                    filtered_X = remover.fit_transform(X)
                    
                    if hasattr(remover, 'outlier_indices_'):
                        outlier_indices = remover.outlier_indices_
                        kept_indices = [i for i in range(len(X)) if i not in outlier_indices]
                    else:
                        kept_indices = list(range(filtered_X.shape[0]))
                    
                    filtered_y = y.iloc[kept_indices].reset_index(drop=True)
                    filtered_X = filtered_X.reset_index(drop=True)
                    
                    st.session_state.current_X = filtered_X
                    st.session_state.y_full = filtered_y
                    
                    st.session_state.preprocessing_parameters['outlier_removal'] = {
                        'threshold': float(threshold),
                        'applied': True,
                        'removed_samples': X.shape[0] - filtered_X.shape[0]
                    }
                    
                    st.session_state.outliers_processed = True
                    
                    st.success(f"Removed {X.shape[0] - filtered_X.shape[0]} outliers")
                    st.info(f"Remaining samples: {filtered_X.shape[0]}")

                    fig, ax = plt.subplots(figsize=(12, 6))
                    for i in range(min(50, filtered_X.shape[0])):
                        ax.plot(x_axis, filtered_X.iloc[i].values, alpha=0.5)
                    ax.set_xlabel("Wavenumber (cm⁻¹)")
                    ax.set_ylabel("Intensity")
                    ax.set_title("Spectra After Outlier Removal")
                    ax.grid(True)
                    st.pyplot(fig)
                    plt.close(fig)
                    
                except Exception as e:
                    st.error(f"Error in outlier removal: {str(e)}")
        else:
            st.session_state.preprocessing_parameters['outlier_removal'] = {'applied': False}
            st.session_state.outliers_processed = True
            st.info("Outlier removal skipped")
        
        if st.session_state.outliers_processed:
            if st.button("Proceed to Train-Test Split"):
                st.session_state.step = 4
                st.rerun()

    elif st.session_state.step == 4:
        st.header("Step 4: Train-Test Split")
        
        if not st.session_state.targets_set:
            st.error("Please complete previous steps first")
            return
        
        X = st.session_state.current_X
        y = st.session_state.y_full
        
        st.subheader("Create Train-Test Split")
        st.info("Now splitting the preprocessed data into training and test sets")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Samples", X.shape[0])
            st.metric("Total Features", X.shape[1])
        with col2:
            st.metric("Target Variables", y.shape[1] if len(y.shape) > 1 else 1)
        
        test_size = st.slider("Select test size fraction", 0.1, 0.5, 0.3, 0.05)
        
        if st.button("Create Train-Test Split"):
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.data_split = True
                
                save_step_data('split', X_train, X_test, y_train, y_test)
                
                st.success("Train-test split created successfully!")
                st.info(f"Training set: {X_train.shape[0]} samples")
                st.info(f"Test set: {X_test.shape[0]} samples")
                st.info(f"Features: {X_train.shape[1]}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Training Set**")
                    st.write(f"X_train: {X_train.shape}")
                    st.write(f"y_train: {y_train.shape}")
                with col2:
                    st.write("**Test Set**")
                    st.write(f"X_test: {X_test.shape}")
                    st.write(f"y_test: {y_test.shape}")
                
            except Exception as e:
                st.error(f"Error creating train-test split: {str(e)}")
        
        if st.session_state.data_split:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Proceed to Wavelet Denoising"):
                    st.session_state.step = 5
                    st.rerun()
            with col2:
                if st.button("Skip to Preprocessing"):
                    st.session_state.skipped_steps.add(5)
                    st.session_state.step = 6
                    st.rerun()

    elif st.session_state.step == 5:
        st.header("Step 5: Wavelet Denoising")

        if not st.session_state.data_split:
            st.error("Please complete train-test split first")
            if st.button("Go back to Train-Test Split"):
                st.session_state.step = 4
                st.rerun()
            return

        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test
        x_axis = st.session_state.x_axis if 'x_axis' in st.session_state else np.arange(X_train.shape[1])

        st.subheader("Original Training Spectra")
        fig, ax = plt.subplots(figsize=(12, 6))
        for i in range(min(50, X_train.shape[0])):
            ax.plot(x_axis, X_train.iloc[i].values, alpha=0.5)
        ax.set_xlabel("Wavenumber (cm⁻¹)")
        ax.set_ylabel("Intensity")
        ax.set_title("Original Training Spectra")
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)

        col1, col2, col3 = st.columns(3)
        with col1:
            wavelet = st.selectbox("Wavelet Type", ["db4", "sym4", "rbio4.4", "coif1"], index=2)
        with col2:
            level = st.slider("Decomposition Level", 1, 5, 3)
        with col3:
            mode = st.radio("Thresholding Mode", ["soft", "hard"], horizontal=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Apply Wavelet Denoising"):
                try:
                    denoiser = WaveletDenoiser(wavelet=wavelet, level=level, threshold_mode=mode)
                    
                    with st.spinner("Fitting wavelet denoiser on training data..."):
                        denoiser.fit(X_train)
                    
                    with st.spinner("Applying wavelet denoising..."):
                        denoised_X_train = denoiser.transform(X_train)
                        denoised_X_test = denoiser.transform(X_test)

                    st.session_state.X_train = denoised_X_train
                    st.session_state.X_test = denoised_X_test
                    st.session_state.fitted_denoiser = denoiser  
                    st.session_state.preprocessing_parameters['wavelet'] = {
                        'wavelet': str(wavelet),
                        'level': int(level),
                        'threshold_mode': str(mode),
                        'fitted_threshold': float(denoiser.fitted_threshold_),
                        'applied': True
                    }
                    st.session_state.wavelet_done = True

                    save_step_data('wavelet', denoised_X_train, denoised_X_test, y_train, y_test)

                    st.subheader("Denoised Training Spectra")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    for i in range(min(50, denoised_X_train.shape[0])):
                        ax.plot(x_axis, denoised_X_train.iloc[i].values, alpha=0.5)
                    ax.set_xlabel("Wavenumber (cm⁻¹)")
                    ax.set_ylabel("Intensity")
                    ax.set_title("Wavelet Denoised Training Spectra")
                    ax.grid(True)
                    st.pyplot(fig)
                    plt.close(fig)

                    st.success(f"Wavelet denoising completed. Training shape: {denoised_X_train.shape}")
                    st.info(f"Fitted threshold: {denoiser.fitted_threshold_:.6f}")

                except Exception as e:
                    st.error(f"Error in wavelet denoising: {str(e)}")

        with col2:
            if st.button("Skip Wavelet Denoising"):
                st.session_state.wavelet_done = False
                st.session_state.fitted_denoiser = None
                st.session_state.preprocessing_parameters['wavelet'] = {'applied': False}
                st.info("Wavelet denoising skipped")
                st.session_state.wavelet_done = True

        if st.session_state.get('wavelet_done'):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Proceed to Preprocessing"):
                    st.session_state.step = 6
                    st.rerun()
            with col2:
                if st.button("Skip to Dimensionality Reduction"):
                    st.session_state.skipped_steps.add(6)
                    st.session_state.step = 7
                    st.rerun()
    
##################################################################################################################################################################

    elif st.session_state.step == 6:
        st.header("Step 6: Spectral Data Preprocessing")

        if not st.session_state.data_split:
            st.error("Please complete train-test split first")
            return
        
        X_train, X_test, y_train, y_test = get_current_data()
        if X_train is None:
            st.error("No training data available")
            return
        
        if 'spectral_data_train' not in st.session_state or st.button("Reset Preprocessing"):
            X_train, X_test, y_train, y_test = get_current_data()
            if X_train is None:
                if hasattr(st.session_state, 'X_train'):
                    X_train = st.session_state.X_train
                    X_test = st.session_state.X_test
                    y_train = st.session_state.y_train
                    y_test = st.session_state.y_test
                else:
                    st.error("No training data available")
                    return
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                X_train.to_csv(tmp.name, index=False)
                st.session_state.spectral_data_train = SpectralData(tmp.name)
                st.session_state.temp_file_train = tmp.name
                
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                X_test.to_csv(tmp.name, index=False)
                st.session_state.spectral_data_test = SpectralData(tmp.name)
                st.session_state.temp_file_test = tmp.name
            

            st.session_state.preprocessing_history = []
            st.session_state.preprocessing_parameters['spectral_steps'] = []
            st.session_state.preprocessing_done = False
            st.session_state.preprocessing_page = 1
            st.session_state.preprocessing_mode = None
            
            st.session_state.fft_applied = False
            st.session_state.opls_applied = False
            st.session_state.advanced_preprocessing_done = False
            
            for key in ['auto_results', 'auto_optimizer', 'selected_technique', 'optimization_running']:
                if key in st.session_state:
                    del st.session_state[key]
            
            st.success("Preprocessing reset successfully")
            st.rerun()
        
        spectral_data_train = st.session_state.spectral_data_train
        spectral_data_test = st.session_state.spectral_data_test
        
        st.markdown("---")
        
        if 'preprocessing_page' not in st.session_state:
            st.session_state.preprocessing_page = 1
        
        st.subheader("Preprocessing Options")
        st.markdown("Before Preprocessing plot of spectra ")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        for i in range(min(100, len(spectral_data_train.spc))):
            ax.plot(spectral_data_train.wav, spectral_data_train.spc.iloc[i], alpha=0.7)
        ax.set_title("Original Training Spectra (Before Trimming)")
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Intensity")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Standard Preprocessing", use_container_width=True, type="primary" if st.session_state.preprocessing_page == 1 else "secondary"):
                st.session_state.preprocessing_page = 1
                st.rerun()
        with col2:
            if st.button("Advanced Preprocessing (FFT & OPLS)", use_container_width=True, type="primary" if st.session_state.preprocessing_page == 2 else "secondary"):
                st.session_state.preprocessing_page = 2
                st.rerun()
        with col3:
            if st.button("Combined Preprocessing", use_container_width=True, type="primary" if st.session_state.preprocessing_page == 3 else "secondary"):
                st.session_state.preprocessing_page = 3
                st.rerun()
        
        st.markdown("---")
        
        if st.session_state.preprocessing_page == 1:
            st.subheader("Standard Preprocessing Methods")
            
            if 'preprocessing_mode' not in st.session_state:
                st.session_state.preprocessing_mode = None
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("AUTOMATED PREPROCESSING\n\nAI-optimized pipeline selection", 
                            type="primary", 
                            use_container_width=True,
                            help="Uses Optuna optimization to find optimal preprocessing pipeline"):
                    st.session_state.preprocessing_mode = "automated"
                    st.rerun()
            
            with col2:
                if st.button("MANUAL PREPROCESSING\n\nCustom technique selection", 
                            type="secondary", 
                            use_container_width=True,
                            help="Manually configure preprocessing techniques"):
                    st.session_state.preprocessing_mode = "manual"
                    st.rerun()
            
            if st.session_state.preprocessing_mode == "automated":
                st.subheader("Automated Preprocessing Optimization")
                
                if st.session_state.preprocessing_history:
                    st.subheader("Preprocessing History")
                    for i, step in enumerate(st.session_state.preprocessing_history):
                        st.write(f"{i+1}. {step}")
                
                st.subheader("Step 1: Trimming (Manual Only)")
                trim_expander = st.expander("Configure Trimming", expanded=False)
                with trim_expander:
                    enable_trimming = st.checkbox("Enable Trimming", key="enable_trimming")
                    
                    if enable_trimming:
                        st.subheader("Original Training Spectra (Before Trimming)")

                        try:
                            fig, ax = plt.subplots(figsize=(10, 5))
                            for i in range(min(100, len(spectral_data_train.spc))):
                                ax.plot(spectral_data_train.wav, spectral_data_train.spc.iloc[i], alpha=0.7)
                            ax.set_title("Original Training Spectra (Before Trimming)")
                            ax.set_xlabel("Wavelength")
                            ax.set_ylabel("Intensity")
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                            plt.close(fig)
                        except Exception as e:
                            st.warning(f"Could not display original spectra plot: {str(e)}")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            trim_type = st.radio("Trim Type:", ["Trim", "Inverse Trim"], key="trim_type")
                        with col2:
                            start = st.number_input("Start Wavelength", 
                                                value=float(spectral_data_train.wav.min()), 
                                                key="trim_start")
                        with col3:
                            end = st.number_input("End Wavelength", 
                                                value=float(spectral_data_train.wav.max()), 
                                                key="trim_end")
                        
                        if st.button("Apply Trimming", type="secondary"):
                            try:
                                if trim_type == "Trim":
                                    spectral_data_train.trim(start=start, end=end)
                                    spectral_data_test.trim(start=start, end=end)
                                    trim_step = f"Trim: {start:.2f} - {end:.2f}"
                                else:
                                    spectral_data_train.invtrim(start=start, end=end)
                                    spectral_data_test.invtrim(start=start, end=end)
                                    trim_step = f"Inverse Trim: {start:.2f} - {end:.2f}"
                                
                                st.session_state.preprocessing_history.append(trim_step)
                                st.session_state.preprocessing_parameters['spectral_steps'].append(trim_step)
                                
                                st.success(f"Applied: {trim_step}")
                                st.info(f"New training shape: {spectral_data_train.spc.shape}")
                                st.info(f"New test shape: {spectral_data_test.spc.shape}")
                                
                                try:
                                    fig, ax = plt.subplots(figsize=(10, 5))
                                    for i in range(min(5, len(spectral_data_train.spc))):
                                        ax.plot(spectral_data_train.wav, spectral_data_train.spc.iloc[i], alpha=0.7)
                                    ax.set_title(f"After {trim_step}")
                                    ax.set_xlabel("Wavelength")
                                    ax.set_ylabel("Intensity")
                                    ax.grid(True, alpha=0.3)
                                    st.pyplot(fig)
                                    plt.close(fig)
                                except:
                                    pass
                                    
                            except Exception as e:
                                st.error(f"Error in trimming: {str(e)}")
                
                if not hasattr(st.session_state, 'y_train') or st.session_state.y_train is None:
                    st.error("Target values are required for automated preprocessing. Please complete target selection first.")
                    return
                
                st.markdown("### Select Spectroscopic Technique")
                technique = st.selectbox(
                    "Choose your analytical technique:",
                    ["Raman Spectroscopy", "NIR Spectroscopy", "FTIR Spectroscopy", "Mass Spectrometry"],
                    help="Select the appropriate technique for optimized preprocessing"
                )
                
                technique_info = {
                    "Raman Spectroscopy": {
                        "description": "Optimized for Raman spectroscopy with baseline correction, smoothing, normalization, and derivative techniques.",
                        "key_features": ["Baseline correction (AsLS, Polyfit, Pearson)", "SNV/MSC normalization", "Savitzky-Golay derivatives", "Multiple centering options"]
                    },
                    "NIR Spectroscopy": {
                        "description": "Specialized for NIR with scatter correction, derivatives, and NIR-specific normalization techniques.",
                        "key_features": ["Advanced scatter correction (SNV, MSC, Detrending)", "1st/2nd order derivatives", "Noise reduction", "Mean centering"]
                    },
                    "FTIR Spectroscopy": {
                        "description": "Tailored for FTIR with peak normalization and FTIR-optimized baseline correction.",
                        "key_features": ["Peak-based normalization", "Area normalization", "Optimized baseline correction", "SG derivatives"]
                    },
                    "Mass Spectrometry": {
                        "description": "Designed for MS data with intensity normalization, background subtraction, and noise reduction.",
                        "key_features": ["Area normalization (TIC equivalent)", "Peak normalization", "Background subtraction", "Specialized smoothing"]
                    }
                }
                
                with st.expander(f"Information: {technique} Details"):
                    info = technique_info[technique]
                    st.write(f"**Description:** {info['description']}")
                    st.write("**Key Features:**")
                    for feature in info['key_features']:
                        st.write(f"• {feature}")
                
                st.markdown("""
                **Fully Automated Process:**
                - Optimizes preprocessing parameters only
                - Tests all models (SVR, XGBoost, Random Forest, PLS, KNN) with default parameters
                - Cross-validation based optimization
                """)
                
                st.subheader("Optimization Settings")
                col1, col2 = st.columns(2)
                with col1:
                    n_trials = st.selectbox("Select number of Trials", [10, 25, 50, 100, 150, 200, 300], index=1)
                with col2:
                    cv_folds = st.selectbox("CV Folds", [3, 5, 7, 10, 15], index=1)
                
                optimization_key = f"auto_opt_{technique}_{n_trials}_{cv_folds}"
                
                if st.button("START AUTOMATED PREPROCESSING", type="primary", use_container_width=True, key=optimization_key):
                    if 'optimization_running' in st.session_state and st.session_state.optimization_running:
                        st.warning("Optimization is already running. Please wait for it to complete.")
                        return
                    
                    st.session_state.optimization_running = True
                    
                    progress_container = st.container()
                    with progress_container:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                    
                    def update_progress(progress, message):
                        try:
                            progress_bar.progress(min(100, max(0, int(progress))) / 100)
                            status_text.text(f"Automated {technique.split()[0]} preprocessing: {message}")
                        except Exception as e:
                            print(f"Progress update error: {e}")
                    
                    try:
                        with st.spinner(f"Initializing automated {technique.split()[0]} preprocessing system..."):
                            X_current_train = spectral_data_train.spc.values
                            X_current_test = spectral_data_test.spc.values
                            y_current = y_train
                            
                            if hasattr(y_current, 'values'):
                                y_data = y_current.values.ravel() if len(y_current.shape) > 1 else y_current.values
                            else:
                                y_data = y_current.ravel() if len(y_current.shape) > 1 else y_current
                            
                            if X_current_train.shape[0] != len(y_data):
                                st.error(f"Data shape mismatch: X has {X_current_train.shape[0]} samples, y has {len(y_data)} samples")
                                st.session_state.optimization_running = False
                                return
                            import numpy as np
                            if np.any(np.isnan(X_current_train)) or np.any(np.isinf(X_current_train)):
                                st.error("X training data contains NaN or infinite values")
                                st.session_state.optimization_running = False
                                return
                            
                            if np.any(np.isnan(y_data)) or np.any(np.isinf(y_data)):
                                st.error("y data contains NaN or infinite values")
                                st.session_state.optimization_running = False
                                return
                            
                            st.info(f"Data prepared: {X_current_train.shape[0]} training samples, {X_current_test.shape[0]} test samples")
                            
                            optimizer = None
                            if technique == "Raman Spectroscopy":
                                optimizer = NIRPreprocessingOptimizer(
                                    X_train=X_current_train,
                                    X_test=X_current_test,
                                    y_train=y_data,
                                    y_test=y_test.values.ravel() if hasattr(y_test, 'values') else y_test,
                                    cv_folds=cv_folds,
                                    n_trials=n_trials,
                                    random_state=42
                                )
                            elif technique == "NIR Spectroscopy":
                                optimizer = NIRPreprocessingOptimizer(
                                    X_train=X_current_train,
                                    X_test=X_current_test,
                                    y_train=y_data,
                                    y_test=y_test.values.ravel() if hasattr(y_test, 'values') else y_test,
                                    cv_folds=cv_folds,
                                    n_trials=n_trials,
                                    random_state=42
                                )
                            
                            elif technique == "FTIR Spectroscopy":
                                st.error("FTIR Spectroscopy optimizer not yet implemented. Please use Raman Spectroscopy for now.")
                                st.session_state.optimization_running = False
                                return
                            elif technique == "Mass Spectrometry":
                                st.error("Mass Spectrometry optimizer not yet implemented. Please use Raman Spectroscopy for now.")
                                st.session_state.optimization_running = False
                                return
                        
                        if optimizer is None:
                            st.session_state.optimization_running = False
                            return
                        
                        st.info(f"Starting optimization with {n_trials} trials and {cv_folds}-fold cross-validation...")
                        results = optimizer.optimize(progress_callback=update_progress)
                        
                        progress_bar.empty()
                        status_text.empty()
                        
                        if not results.get('success', False):
                            st.error(f"Automated {technique} preprocessing failed: {results.get('error', 'Unknown error')}")
                            st.session_state.optimization_running = False
                            return
                        
                        X_processed_train = optimizer.apply_best_preprocessing(X_current_train, fit_mode=False)
                        X_processed_test = optimizer.apply_best_preprocessing(X_current_test, fit_mode=False)
                        
                        processed_spectral_data_train = copy.deepcopy(spectral_data_train)
                        processed_spectral_data_test = copy.deepcopy(spectral_data_test)
                        
                        n_features = X_processed_train.shape[1]
                        original_columns = processed_spectral_data_train.spc.columns
                        
                        if n_features <= len(original_columns):
                            processed_spectral_data_train.spc = pd.DataFrame(X_processed_train, columns=original_columns[:n_features])
                            processed_spectral_data_test.spc = pd.DataFrame(X_processed_test, columns=original_columns[:n_features])
                        else:
                            new_columns = [f"feature_{i}" for i in range(n_features)]
                            processed_spectral_data_train.spc = pd.DataFrame(X_processed_train, columns=new_columns)
                            processed_spectral_data_test.spc = pd.DataFrame(X_processed_test, columns=new_columns)
                        
                        if X_processed_train.shape[1] != len(processed_spectral_data_train.wav):
                            if X_processed_train.shape[1] <= len(processed_spectral_data_train.wav):
                                processed_spectral_data_train.wav = processed_spectral_data_train.wav[:X_processed_train.shape[1]]
                                processed_spectral_data_test.wav = processed_spectral_data_test.wav[:X_processed_test.shape[1]]
                            else:
                                new_wav = np.arange(X_processed_train.shape[1])
                                processed_spectral_data_train.wav = pd.Series(new_wav)
                                processed_spectral_data_test.wav = pd.Series(new_wav)
                        
                        st.session_state.auto_results = results
                        st.session_state.auto_optimizer = optimizer
                        st.session_state.selected_technique = technique
                        
                        st.session_state.spectral_data_train = processed_spectral_data_train
                        st.session_state.spectral_data_test = processed_spectral_data_test
                        
                        st.session_state.X_train = processed_spectral_data_train.spc.copy()
                        st.session_state.X_test = processed_spectral_data_test.spc.copy()
                        
                        save_step_data('preprocessing', 
                                     processed_spectral_data_train.spc.copy(),
                                     processed_spectral_data_test.spc.copy(),
                                     y_train, y_test)
                        
                        technique_short = technique.split()[0]
                        automated_steps = [f"Auto-{technique_short}-{i+1}: {step['method']}" 
                                        for i, step in enumerate(results['best_pipeline'])]
                        st.session_state.preprocessing_history.extend(automated_steps)
                        st.session_state.preprocessing_parameters['spectral_steps'].extend(automated_steps)
                        
                        st.session_state.preprocessing_parameters['automated_technique'] = technique
                        st.session_state.preprocessing_parameters['automated_pipeline'] = results['best_pipeline']
                        st.session_state.preprocessing_parameters['automated_optimizer_info'] = {
                            'technique': technique,
                            'cv_folds': cv_folds,
                            'n_trials': n_trials,
                            'best_pipeline': results['best_pipeline'],
                            'best_params': results['best_params'],
                            'all_model_results': results['all_model_results']
                        }
                        
                        st.success(f"Automated {technique} preprocessing completed! Best CV R² score: {results['cv_score']:.4f}")
                        
                        st.subheader(f"Automated {technique} Preprocessing Results")
                        
                        st.write("**Best CV Performance:**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("CV R² Score", f"{results['cv_score']:.4f}")
                        with col2:
                            st.metric("CV RMSE", f"{results.get('cv_rmse', 0):.4f}")
                        with col3:
                            st.metric("Trials Completed", n_trials)
                        
                        st.write("**Optimal Preprocessing Pipeline:**")
                        for i, step in enumerate(results['best_pipeline']):
                            st.write(f"{i+1}. {step['method']} - {step.get('params', {})}")
                        
                        if 'all_model_results' in results:
                            st.write("**Model Comparison Results:**")
                            model_results_df = pd.DataFrame(results['all_model_results'])
                            st.dataframe(model_results_df)
                        
                        with st.expander("View Best Parameters"):
                            st.json(results['best_params'])
                        
                        try:
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                            
                            for i in range(min(10, len(spectral_data_train.spc))):
                                ax1.plot(spectral_data_train.wav, spectral_data_train.spc.iloc[i], alpha=0.6)
                            ax1.set_title("Before Preprocessing")
                            ax1.set_xlabel("Wavelength")
                            ax1.set_ylabel("Intensity")
                            ax1.grid(True, alpha=0.3)
                            
                            for i in range(min(10, len(processed_spectral_data_train.spc))):
                                ax2.plot(processed_spectral_data_train.wav, processed_spectral_data_train.spc.iloc[i], alpha=0.6)
                            ax2.set_title("After Automated Preprocessing")
                            ax2.set_xlabel("Wavelength")
                            ax2.set_ylabel("Intensity")
                            ax2.grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                        except Exception as plot_error:
                            st.warning(f"Could not create comparison plot: {plot_error}")
                        
                        st.session_state.preprocessing_done = True
                        
                        st.info("Automated preprocessing completed! You can now proceed to model training or apply additional manual preprocessing steps.")
                        
                    except Exception as e:
                        st.error(f"Automated preprocessing system failed: {str(e)}")
                        import traceback
                        with st.expander("Error Details"):
                            st.text(traceback.format_exc())
                    
                    finally:
                        st.session_state.optimization_running = False
                        try:
                            progress_bar.empty()
                            status_text.empty()
                        except:
                            pass
            
            elif st.session_state.preprocessing_mode == "manual":
                st.subheader("Manual Preprocessing Configuration")
                
                selected_techniques = st.multiselect(
                    "Choose preprocessing techniques (applied in order, can repeat):", 
                    ['Trim', 'Baseline Correction', 'Smoothing', 'Normalization', 'Center', 'Derivative', 'SG Derivative'],
                    key="selected_preprocessing_techniques"
                )

                processed_steps = []
                current_params = {}
                
                if 'spectral_parameters' not in st.session_state.preprocessing_parameters:
                    st.session_state.preprocessing_parameters['spectral_parameters'] = {}
                
                for idx, technique in enumerate(selected_techniques):
                    st.markdown(f"### {technique} (Step {idx + 1})")
                    
                    if technique == 'Trim':
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            trim_type = st.radio(f"Trim Type (Step {idx + 1}):", ["Trim", "Inverse Trim"], key=f"trim_type_{idx}")
                        with col2:
                            start = st.number_input(f"Start Wavelength (Step {idx + 1})", 
                                                value=float(spectral_data_train.wav.min()), 
                                                key=f"trim_start_{idx}")
                        with col3:
                            end = st.number_input(f"End Wavelength (Step {idx + 1})", 
                                                value=float(spectral_data_train.wav.max()), 
                                                key=f"trim_end_{idx}")
                        
                        current_params[f'trim_{idx}'] = {
                            'type': trim_type,
                            'start': start,
                            'end': end
                        }
                    
                    elif technique == 'Baseline Correction':
                        baseline_methods = st.multiselect(f"Baseline Correction Methods (Step {idx + 1}):", 
                                                        ["AsLS", "Polyfit", "Pearson"], 
                                                        key=f"baseline_methods_{idx}")
                        
                        baseline_params = {}
                        for method in baseline_methods:
                            st.markdown(f"#### {method} Parameters")
                            if method == "AsLS":
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    lam = st.number_input(f"Lambda (Step {idx + 1}, {method})", 
                                                        value=1000000, key=f"asls_lam_{idx}")
                                with col2:
                                    p = st.number_input(f"p (Step {idx + 1}, {method})", 
                                                      value=0.001, key=f"asls_p_{idx}")
                                with col3:
                                    niter = st.number_input(f"Iterations (Step {idx + 1}, {method})", 
                                                          value=10, key=f"asls_niter_{idx}")
                                baseline_params[f'AsLS_{idx}'] = {'lam': lam, 'p': p, 'niter': niter}
                            
                            elif method == "Polyfit":
                                col1, col2 = st.columns(2)
                                with col1:
                                    order = st.number_input(f"Polynomial Order (Step {idx + 1}, {method})", 
                                                          value=3, key=f"poly_order_{idx}")
                                with col2:
                                    niter = st.number_input(f"Iterations (Step {idx + 1}, {method})", 
                                                          value=3, key=f"poly_niter_{idx}")
                                baseline_params[f'Polyfit_{idx}'] = {'order': order, 'niter': niter}
                            
                            elif method == "Pearson":
                                col1, col2 = st.columns(2)
                                with col1:
                                    u = st.number_input(f"u Parameter (Step {idx + 1}, {method})", 
                                                      value=10, key=f"pearson_u_{idx}")
                                with col2:
                                    v = st.number_input(f"v Parameter (Step {idx + 1}, {method})", 
                                                      value=10, key=f"pearson_v_{idx}")
                                baseline_params[f'Pearson_{idx}'] = {'u': u, 'v': v}
                        
                        current_params[f'baseline_{idx}'] = {
                            'methods': baseline_methods,
                            'parameters': baseline_params
                        }
                    
                    elif technique == 'Smoothing':
                        smoothing_methods = st.multiselect(f"Smoothing Methods (Step {idx + 1}):", 
                                                         ["Rolling", "Savitzky-Golay"], 
                                                         key=f"smoothing_methods_{idx}")
                        
                        smoothing_params = {}
                        for method in smoothing_methods:
                            st.markdown(f"#### {method} Parameters")
                            if method == "Rolling":
                                window = st.number_input(f"Window Size (Step {idx + 1}, {method})", 
                                                       value=5, key=f"rolling_window_{idx}")
                                smoothing_params[f'Rolling_{idx}'] = {'window': window}
                            
                            elif method == "Savitzky-Golay":
                                col1, col2 = st.columns(2)
                                with col1:
                                    window = st.number_input(f"Window Size (Step {idx + 1}, {method})", 
                                                           value=5, key=f"sg_window_{idx}")
                                with col2:
                                    poly = st.number_input(f"Polynomial Order (Step {idx + 1}, {method})", 
                                                         value=3, key=f"sg_poly_{idx}")
                                smoothing_params[f'SG_{idx}'] = {'window': window, 'poly': poly}
                        
                        current_params[f'smoothing_{idx}'] = {
                            'methods': smoothing_methods,
                            'parameters': smoothing_params
                        }
                    
                    elif technique == 'Normalization':
                        normalization_methods = st.multiselect(f"Normalization Methods (Step {idx + 1}):", 
                                                             ["SNV", "MSC", "Detrend", "Area", "Peak Normalization", "Vector", "Min-max", "Pareto"], 
                                                             key=f"normalization_methods_{idx}")
                        
                        normalization_params = {}
                        for method in normalization_methods:
                            if method == "Detrend":
                                order = st.number_input(f"Detrend Order (Step {idx + 1})", 
                                                      value=2, key=f"detrend_order_{idx}")
                                normalization_params[f'Detrend_{idx}'] = {'order': order}
                            elif method == "Peak Normalization":
                                wave = st.number_input(f"Peak Wavenumber (Step {idx + 1})", 
                                                     value=float(spectral_data_train.wav.median()), 
                                                     key=f"peak_wave_{idx}")
                                normalization_params[f'Peak_{idx}'] = {'wave': wave}
                            elif method == "Min-max":
                                col1, col2 = st.columns(2)
                                with col1:
                                    minv = st.number_input(f"Min Value (Step {idx + 1})", 
                                                         value=0, key=f"minmax_min_{idx}")
                                with col2:
                                    maxv = st.number_input(f"Max Value (Step {idx + 1})", 
                                                         value=1, key=f"minmax_max_{idx}")
                                normalization_params[f'Minmax_{idx}'] = {'minv': minv, 'maxv': maxv}
                        
                        current_params[f'normalization_{idx}'] = {
                            'methods': normalization_methods,
                            'parameters': normalization_params
                        }
                    
                    elif technique == 'Center':
                        center_methods = st.multiselect(f"Centering Methods (Step {idx + 1}):", 
                                                      ["Mean (spectrum)", "Mean (wavelength)", "Last Point"], 
                                                      key=f"center_methods_{idx}")
                        
                        current_params[f'center_{idx}'] = {
                            'methods': center_methods
                        }
                    
                    elif technique == 'Derivative':
                        derivative_options = st.multiselect(f"Derivative Options (Step {idx + 1}):", 
                                                          ["Subtract", "Reset"], 
                                                          key=f"derivative_options_{idx}")
                        
                        derivative_params = {}
                        for option in derivative_options:
                            if option == "Subtract":
                                subtract_idx = st.number_input(f"Subtract Index (Step {idx + 1})", 
                                                             value=0, key=f"subtract_idx_{idx}")
                                derivative_params['subtract_idx'] = subtract_idx
                        
                        current_params[f'derivative_{idx}'] = {
                            'options': derivative_options,
                            'parameters': derivative_params
                        }
                    
                    elif technique == 'SG Derivative':
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            window = st.number_input(f"Window Size (Step {idx + 1})", 
                                                   value=5, key=f"sgderiv_window_{idx}")
                        with col2:
                            poly = st.number_input(f"Polynomial Order (Step {idx + 1})", 
                                                 value=3, key=f"sgderiv_poly_{idx}")
                        with col3:
                            order = st.number_input(f"Derivative Order (Step {idx + 1})", 
                                                  value=1, key=f"sgderiv_order_{idx}")
                        
                        current_params[f'sg_derivative_{idx}'] = {
                            'window': window,
                            'poly': poly,
                            'order': order
                        }
                
                if selected_techniques and st.button("Apply Manual Preprocessing"):
                    try:
                        st.session_state.preprocessing_parameters['spectral_parameters'] = current_params
                        
                        applied_steps = []
                        all_successful = True
                        
                        for idx, technique in enumerate(selected_techniques):
                            st.write(f"Applying {technique} (Step {idx + 1})...")
                            
                            try:
                                if technique == 'Trim':
                                    params = current_params[f'trim_{idx}']
                                    if params['type'] == "Trim":
                                        spectral_data_train.trim(start=params['start'], end=params['end'])
                                        spectral_data_test.trim(start=params['start'], end=params['end'])
                                        step_name = f"Trim: {params['start']:.2f} - {params['end']:.2f}"
                                    else:
                                        spectral_data_train.invtrim(start=params['start'], end=params['end'])
                                        spectral_data_test.invtrim(start=params['start'], end=params['end'])
                                        step_name = f"Inverse Trim: {params['start']:.2f} - {params['end']:.2f}"
                                    applied_steps.append(step_name)
                                
                                elif technique == 'Baseline Correction':
                                    params = current_params[f'baseline_{idx}']
                                    for method in params['methods']:
                                        if method == "AsLS":
                                            p = params['parameters'][f'AsLS_{idx}']
                                            spectral_data_train.AsLS(lam=p['lam'], p=p['p'], niter=int(p['niter']))
                                            spectral_data_test.AsLS(lam=p['lam'], p=p['p'], niter=int(p['niter']))
                                        elif method == "Polyfit":
                                            p = params['parameters'][f'Polyfit_{idx}']
                                            spectral_data_train.polyfit(order=int(p['order']), niter=int(p['niter']))
                                            spectral_data_test.polyfit(order=int(p['order']), niter=int(p['niter']))
                                        elif method == "Pearson":
                                            p = params['parameters'][f'Pearson_{idx}']
                                            spectral_data_train.pearson(u=int(p['u']), v=int(p['v']))
                                            spectral_data_test.pearson(u=int(p['u']), v=int(p['v']))
                                        applied_steps.append(f"Baseline Correction: {method}")
                                
                                elif technique == 'Smoothing':
                                    params = current_params[f'smoothing_{idx}']
                                    for method in params['methods']:
                                        if method == "Rolling":
                                            p = params['parameters'][f'Rolling_{idx}']
                                            spectral_data_train.rolling(window=int(p['window']))
                                            spectral_data_test.rolling(window=int(p['window']))
                                        elif method == "Savitzky-Golay":
                                            p = params['parameters'][f'SG_{idx}']
                                            spectral_data_train.SGSmooth(window=int(p['window']), poly=int(p['poly']))
                                            spectral_data_test.SGSmooth(window=int(p['window']), poly=int(p['poly']))
                                        applied_steps.append(f"Smoothing: {method}")
                                
                                elif technique == 'Normalization':
                                    params = current_params[f'normalization_{idx}']
                                    for method in params['methods']:
                                        if method == "SNV":
                                            spectral_data_train.snv()
                                            spectral_data_test.snv()
                                        elif method == "MSC":
                                            spectral_data_train.msc()
                                            spectral_data_test.msc()
                                        elif method == "Detrend":
                                            p = params['parameters'][f'Detrend_{idx}']
                                            spectral_data_train.detrend(order=p['order'])
                                            spectral_data_test.detrend(order=p['order'])
                                        elif method == "Area":
                                            spectral_data_train.area()
                                            spectral_data_test.area()
                                        elif method == "Peak Normalization":
                                            p = params['parameters'][f'Peak_{idx}']
                                            spectral_data_train.peaknorm(wavenumber=p['wave'])
                                            spectral_data_test.peaknorm(wavenumber=p['wave'])
                                        elif method == "Vector":
                                            spectral_data_train.vector()
                                            spectral_data_test.vector()
                                        elif method == "Min-max":
                                            p = params['parameters'][f'Minmax_{idx}']
                                            spectral_data_train.minmax(min_val=p['minv'], max_val=p['maxv'])
                                            spectral_data_test.minmax(min_val=p['minv'], max_val=p['maxv'])
                                        elif method == "Pareto":
                                            spectral_data_train.pareto()
                                            spectral_data_test.pareto()
                                        applied_steps.append(f"Normalization: {method}")
                                
                                elif technique == 'Center':
                                    params = current_params[f'center_{idx}']
                                    for method in params['methods']:
                                        if method == 'Mean (spectrum)':
                                            spectral_data_train.mean_center(option=False)
                                            spectral_data_test.mean_center(option=False)
                                        elif method == 'Mean (wavelength)':
                                            spectral_data_train.mean_center(option=True)
                                            spectral_data_test.mean_center(option=True)
                                        elif method == 'Last Point':
                                            spectral_data_train.lastpoint()
                                            spectral_data_test.lastpoint()
                                        applied_steps.append(f"Center: {method}")
                                
                                elif technique == 'Derivative':
                                    params = current_params[f'derivative_{idx}']
                                    for option in params['options']:
                                        if option == "Subtract":
                                            spectral_data_train.subtract(spectra=params['parameters']['subtract_idx'])
                                            spectral_data_test.subtract(spectra=params['parameters']['subtract_idx'])
                                        elif option == "Reset":
                                            spectral_data_train.reset()
                                            spectral_data_test.reset()
                                        applied_steps.append(f"Derivative: {option}")
                                
                                elif technique == 'SG Derivative':
                                    params = current_params[f'sg_derivative_{idx}']
                                    spectral_data_train.SGDeriv(
                                        window=int(params['window']), 
                                        poly=int(params['poly']), 
                                        order=int(params['order'])
                                    )
                                    spectral_data_test.SGDeriv(
                                        window=int(params['window']), 
                                        poly=int(params['poly']), 
                                        order=int(params['order'])
                                    )
                                    applied_steps.append(f"SG Derivative: order={params['order']}")
                                
                                st.success(f"  {technique} applied successfully")
                                
                            except Exception as technique_error:
                                st.error(f"  Error applying {technique}: {str(technique_error)}")
                                all_successful = False
                                break
                        
                        if all_successful:
                            st.session_state.X_train = spectral_data_train.spc.copy()
                            st.session_state.X_test = spectral_data_test.spc.copy()
                            
                            st.session_state.preprocessing_history.extend(applied_steps)
                            st.session_state.preprocessing_parameters['spectral_steps'].extend(applied_steps)
                            
                            save_step_data('preprocessing', 
                                         spectral_data_train.spc.copy(),
                                         spectral_data_test.spc.copy(),
                                         y_train, y_test)
                            
                            st.session_state.preprocessing_done = True
                            
                            st.success("Manual preprocessing completed successfully!")
                            st.info(f"Training data shape: {spectral_data_train.spc.shape}")
                            st.info(f"Test data shape: {spectral_data_test.spc.shape}")
                            
                            try:
                                fig, ax = plt.subplots(figsize=(12, 6))
                                
                                for i in range(min(100, len(spectral_data_train.spc))):
                                    ax.plot(spectral_data_train.wav, spectral_data_train.spc.iloc[i], alpha=0.6)
                                ax.set_title("Processed Training Spectra")
                                ax.set_xlabel("Wavelength")
                                ax.set_ylabel("Intensity")
                                ax.grid(True, alpha=0.3)
                                st.pyplot(fig)
                                plt.close(fig)
                            except Exception as plot_error:
                                st.warning(f"Could not create processed spectra plot: {plot_error}")
                        else:
                            st.warning("Some preprocessing steps failed. Please check the parameters and try again.")
                        
                    except Exception as e:
                        st.error(f"Error in manual preprocessing: {str(e)}")
                        import traceback
                        with st.expander("Error Details"):
                            st.text(traceback.format_exc())

        elif st.session_state.preprocessing_page == 2:
            st.subheader("Advanced Preprocessing Methods")
    
            if 'advanced_preprocessing_done' not in st.session_state:
                st.session_state.advanced_preprocessing_done = False
            if 'fft_applied' not in st.session_state:
                st.session_state.fft_applied = False
            if 'opls_applied' not in st.session_state:
                st.session_state.opls_applied = False
            
            current_X_train = st.session_state.X_train if hasattr(st.session_state, 'X_train') else X_train
            current_X_test = st.session_state.X_test if hasattr(st.session_state, 'X_test') else X_test
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Current Training Samples", current_X_train.shape[0])
                st.metric("Current Features", current_X_train.shape[1])
            with col2:
                st.metric("Current Test Samples", current_X_test.shape[0])
                if st.session_state.fft_applied:
                    st.success("FFT Applied")
                if st.session_state.opls_applied:
                    st.success(" OPLS Applied")
            
            st.markdown("---")
            st.subheader("FFT (Fast Fourier Transform) Filtering")
            
            with st.expander("About FFT Filtering", expanded=False):
                st.markdown("""
                **FFT Filtering** removes high-frequency noise from spectral data:
                - **Purpose**: Noise reduction and signal smoothing
                - **Method**: Filters out frequencies above a specified threshold
                - **Best for**: Noisy spectral data with high-frequency artifacts
                - **Effect**: Produces cleaner, smoother spectra
                """)
            
            fft_col1, fft_col2, fft_col3 = st.columns(3)
            with fft_col1:
                fft_threshold = st.number_input(
                    "Frequency Threshold", 
                    value=1000.0, min_value=1.0, max_value=10000.0,
                    help="Frequencies above this threshold will be filtered out"
                )
            with fft_col2:
                fft_sampling_interval = st.number_input(
                    "Sampling Interval", 
                    value=0.02, min_value=0.001, max_value=1.0, format="%.4f",
                    help="Sampling interval for FFT processing"
                )
            with fft_col3:
                show_fft_comparison = st.checkbox("Show Before/After Plot", True, key="fft_comparison")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Apply FFT Filtering", type="primary", use_container_width=True):
                    try:
                        with st.spinner("Applying FFT filtering..."):
                            fft_processor = FFTProcessor(
                                current_X_train, current_X_test, y_train, y_test,
                                threshold=fft_threshold,
                                sampling_interval=fft_sampling_interval, 
                            )
                            
                            fft_processor.fit()
                            fft_train_filtered, fft_test_filtered = fft_processor.transform()
                            
                            fft_train_df = pd.DataFrame(
                                fft_train_filtered,
                                columns=current_X_train.columns if fft_train_filtered.shape[1] == current_X_train.shape[1] else [f'feature_{i}' for i in range(fft_train_filtered.shape[1])],
                                index=current_X_train.index
                            )
                            fft_test_df = pd.DataFrame(
                                fft_test_filtered,
                                columns=current_X_test.columns if fft_test_filtered.shape[1] == current_X_test.shape[1] else [f'feature_{i}' for i in range(fft_test_filtered.shape[1])],
                                index=current_X_test.index
                            )
                            
                            st.session_state.X_train = fft_train_df
                            st.session_state.X_test = fft_test_df
                            st.session_state.fft_applied = True
                            
                            save_step_data('fft_preprocessing', fft_train_df, fft_test_df, y_train, y_test)
                            
                            st.session_state.preprocessing_parameters['fft'] = {
                                'threshold': float(fft_threshold),
                                'sampling_interval': float(fft_sampling_interval),
                                'applied': True
                            }
                            st.session_state.preprocessing_history.append(f"FFT Filtering: threshold={fft_threshold:.0f}")
                            
                            st.success("FFT filtering applied successfully!")
                            st.info(f"Data shape: {current_X_train.shape} → {fft_train_df.shape}")
                            
                            if show_fft_comparison:
                                try:
                                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                                    
                                    sample_idx = min(5, current_X_train.shape[0] - 1)
                                    ax1.plot(current_X_train.iloc[sample_idx].values, 'b-', alpha=0.7, label='Original', linewidth=2)
                                    ax1.plot(fft_train_df.iloc[sample_idx].values, 'r-', alpha=0.8, label='FFT Filtered', linewidth=2)
                                    ax1.set_title("Sample Spectrum: Before vs After FFT")
                                    ax1.set_xlabel("Feature Index")
                                    ax1.set_ylabel("Intensity")
                                    ax1.legend()
                                    ax1.grid(True, alpha=0.3)
                                    
                                    n_plot = min(10, fft_train_df.shape[0])
                                    for i in range(n_plot):
                                        ax2.plot(fft_train_df.iloc[i].values, alpha=0.6)
                                    ax2.set_title("FFT Filtered Training Spectra")
                                    ax2.set_xlabel("Feature Index")
                                    ax2.set_ylabel("Intensity")
                                    ax2.grid(True, alpha=0.3)
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.close(fig)
                                    
                                except Exception as plot_error:
                                    st.warning(f"Could not create comparison plot: {plot_error}")
                                
                    except Exception as e:
                        st.error(f"Error in FFT filtering: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            
            with col2:
                if st.session_state.fft_applied and st.button("Reset FFT", type="secondary", use_container_width=True):
                    if 'preprocessing' in st.session_state.data_history:
                        X_train_pre, X_test_pre, y_train_pre, y_test_pre = get_step_data('preprocessing')
                        if X_train_pre is not None:
                            st.session_state.X_train = X_train_pre
                            st.session_state.X_test = X_test_pre
                        else:
                            st.session_state.X_train = X_train
                            st.session_state.X_test = X_test
                    else:
                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                    
                    st.session_state.fft_applied = False
                    if 'fft' in st.session_state.preprocessing_parameters:
                        del st.session_state.preprocessing_parameters['fft']
                    st.success("FFT filtering reset")
                    st.rerun()
            
            st.markdown("---")
            st.subheader(" OPLS (Orthogonal Partial Least Squares) Analysis")

            with st.expander("About OPLS Analysis", expanded=False):
                st.markdown("""
                **OPLS Analysis** removes variation orthogonal to the target variable:
                - **Purpose**: Filters out systematic variation not related to Y
                - **Method**: Separates predictive and orthogonal components
                - **Best for**: Spectral data with systematic but irrelevant variation
                - **Effect**: Improves model interpretability and reduces overfitting
                """)

            if not hasattr(st.session_state, 'y_train') or st.session_state.y_train is None:
                st.warning("OPLS requires target variables. Please ensure targets are set.")
            else:
                current_X_train = st.session_state.X_train if hasattr(st.session_state, 'X_train') and st.session_state.X_train is not None else X_train
                current_X_test = st.session_state.X_test if hasattr(st.session_state, 'X_test') and st.session_state.X_test is not None else X_test
                y_train = st.session_state.y_train
                y_test = st.session_state.y_test
                
                opls_col1, opls_col2 = st.columns(2)
                with opls_col1:
                    n_opls_components = st.slider(
                        "Number of Orthogonal Components", 
                        1, min(20, current_X_train.shape[1]), 5,
                        help="Number of orthogonal components to filter out"
                    )
                with opls_col2:
                    opls_scale_data = st.checkbox("Scale Data", True, help="Whether to scale the data before OPLS")
                
                opls_col1, opls_col2 = st.columns(2)
                with opls_col1:
                    show_opls_comparison = st.checkbox("Show Before/After Plot", True, key="opls_comparison")
                with opls_col2:
                    show_opls_score = st.checkbox("Show R²X Score", True, key="opls_score")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Apply OPLS Analysis", type="primary", use_container_width=True):
                        try:
                            with st.spinner("Applying OPLS analysis..."):
                                opls_X_train = st.session_state.X_train if hasattr(st.session_state, 'fft_applied') and st.session_state.fft_applied else current_X_train
                                opls_X_test = st.session_state.X_test if hasattr(st.session_state, 'fft_applied') and st.session_state.fft_applied else current_X_test
                                
                                opls = OPLS(opls_X_train.values, opls_X_test.values, y_train, y_test, n_components=n_opls_components, scale=opls_scale_data)
                                
                                opls.fit()
                                opls_train_filtered = opls.transform(opls_X_train.values)
                                opls_test_filtered = opls.transform(opls_X_test.values)
                                
                                opls_train_df = pd.DataFrame(
                                    opls_train_filtered,
                                    columns=opls_X_train.columns if opls_train_filtered.shape[1] == opls_X_train.shape[1] else [f'feature_{i}' for i in range(opls_train_filtered.shape[1])],
                                    index=opls_X_train.index
                                )
                                opls_test_df = pd.DataFrame(
                                    opls_test_filtered,
                                    columns=opls_X_test.columns if opls_test_filtered.shape[1] == opls_X_test.shape[1] else [f'feature_{i}' for i in range(opls_test_filtered.shape[1])],
                                    index=opls_X_test.index
                                )
                                
                                st.session_state.X_train = opls_train_df
                                st.session_state.X_test = opls_test_df
                                if not hasattr(st.session_state, 'opls_applied'):
                                    st.session_state.opls_applied = False
                                st.session_state.opls_applied = True
                                
                                if 'save_step_data' in globals():
                                    save_step_data('opls_preprocessing', opls_train_df, opls_test_df, y_train, y_test)
                                
                                if not hasattr(st.session_state, 'preprocessing_parameters'):
                                    st.session_state.preprocessing_parameters = {}
                                st.session_state.preprocessing_parameters['opls'] = {
                                    'n_components': int(n_opls_components),
                                    'scale': bool(opls_scale_data),
                                    'applied': True
                                }
                                if not hasattr(st.session_state, 'preprocessing_history'):
                                    st.session_state.preprocessing_history = []
                                st.session_state.preprocessing_history.append(f"OPLS: {n_opls_components} orthogonal components filtered")
                                
                                r2x_score = opls.score(opls_X_train.values)
                                
                                st.success(f"OPLS analysis completed! Filtered {n_opls_components} orthogonal components")
                                st.info(f"Data shape: {opls_X_train.shape} → {opls_train_df.shape}")
                                
                                if show_opls_score:
                                    st.metric("R²X Score", f"{r2x_score:.4f}", help="Amount of variation in X explained by filtered X (lower = more orthogonal variation removed)")
                                
                                if show_opls_comparison:
                                    try:
                                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                                        
                                        sample_idx = min(5, opls_X_train.shape[0] - 1)
                                        ax1.plot(opls_X_train.iloc[sample_idx].values, 'b-', alpha=0.7, label='Before OPLS', linewidth=2)
                                        ax1.plot(opls_train_df.iloc[sample_idx].values, 'r-', alpha=0.8, label='After OPLS', linewidth=2)
                                        ax1.set_title("Sample Spectrum: Before vs After OPLS")
                                        ax1.set_xlabel("Feature Index")
                                        ax1.set_ylabel("Intensity")
                                        ax1.legend()
                                        ax1.grid(True, alpha=0.3)
                                        
                                        n_plot = min(10, opls_train_df.shape[0])
                                        for i in range(n_plot):
                                            ax2.plot(opls_train_df.iloc[i].values, alpha=0.6)
                                        ax2.set_title("OPLS Filtered Training Spectra")
                                        ax2.set_xlabel("Feature Index")
                                        ax2.set_ylabel("Intensity")
                                        ax2.grid(True, alpha=0.3)
                                        
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                        plt.close(fig)
                                        
                                    except Exception as plot_error:
                                        st.warning(f"Could not create comparison plot: {plot_error}")
                                
                        except Exception as e:
                            st.error(f"Error in OPLS analysis: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                
                with col2:
                    if hasattr(st.session_state, 'opls_applied') and st.session_state.opls_applied and st.button("Reset OPLS", type="secondary", use_container_width=True):
                        if hasattr(st.session_state, 'fft_applied') and st.session_state.fft_applied and 'get_step_data' in globals() and hasattr(st.session_state, 'data_history') and 'fft_preprocessing' in st.session_state.data_history:
                            X_train_pre, X_test_pre, y_train_pre, y_test_pre = get_step_data('fft_preprocessing')
                            st.session_state.X_train = X_train_pre
                            st.session_state.X_test = X_test_pre
                        elif 'get_step_data' in globals() and hasattr(st.session_state, 'data_history') and 'preprocessing' in st.session_state.data_history:
                            X_train_pre, X_test_pre, y_train_pre, y_test_pre = get_step_data('preprocessing')
                            st.session_state.X_train = X_train_pre if X_train_pre is not None else X_train
                            st.session_state.X_test = X_test_pre if X_test_pre is not None else X_test
                        else:
                            st.session_state.X_train = X_train
                            st.session_state.X_test = X_test
                        
                        st.session_state.opls_applied = False
                        if hasattr(st.session_state, 'preprocessing_parameters') and 'opls' in st.session_state.preprocessing_parameters:
                            del st.session_state.preprocessing_parameters['opls']
                        st.success("OPLS analysis reset")
                        st.rerun()

            st.markdown("---")
            if (hasattr(st.session_state, 'fft_applied') and st.session_state.fft_applied) or (hasattr(st.session_state, 'opls_applied') and st.session_state.opls_applied):
                st.success("Advanced preprocessing applied!")
                
                current_train = st.session_state.X_train
                current_test = st.session_state.X_test
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Final Training Shape", f"{current_train.shape[0]} × {current_train.shape[1]}")
                with col2:
                    st.metric("Final Test Shape", f"{current_test.shape[0]} × {current_test.shape[1]}")
                with col3:
                    applied_methods = []
                    if hasattr(st.session_state, 'fft_applied') and st.session_state.fft_applied:
                        applied_methods.append("FFT")
                    if hasattr(st.session_state, 'opls_applied') and st.session_state.opls_applied:
                        applied_methods.append("OPLS")
                    st.metric("Applied Methods", " + ".join(applied_methods))
                
                if 'save_step_data' in globals():
                    save_step_data('advanced_preprocessing', current_train, current_test, y_train, y_test)
                if not hasattr(st.session_state, 'advanced_preprocessing_done'):
                    st.session_state.advanced_preprocessing_done = False
                st.session_state.advanced_preprocessing_done = True
            else:
                st.info("No advanced preprocessing methods applied yet.")

        elif st.session_state.preprocessing_page == 3:
            st.subheader("Combined Preprocessing (Manual + Automated + Advanced)")
            
            current_X_train = st.session_state.X_train if hasattr(st.session_state, 'X_train') else X_train
            current_X_test = st.session_state.X_test if hasattr(st.session_state, 'X_test') else X_test
            
            st.info("Apply preprocessing methods in sequence: Manual → Automated → Advanced")
            
            steps_applied = []
            if st.session_state.get('preprocessing_done', False):
                steps_applied.append("Standard")
            if st.session_state.get('fft_applied', False):
                steps_applied.append("FFT")
            if st.session_state.get('opls_applied', False):
                steps_applied.append("OPLS")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Current Data Shape", f"{current_X_train.shape[0]} × {current_X_train.shape[1]}")
            with col2:
                if steps_applied:
                    st.success(f"Applied: {', '.join(steps_applied)}")
                else:
                    st.info("No preprocessing applied yet")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Go to Manual", type="secondary", use_container_width=True):
                    st.session_state.preprocessing_page = 1
                    st.session_state.preprocessing_mode = "manual"
                    st.rerun()
            with col2:
                if st.button("Go to Automated", type="secondary", use_container_width=True):
                    st.session_state.preprocessing_page = 1
                    st.session_state.preprocessing_mode = "automated"
                    st.rerun()
            with col3:
                if st.button("Go to Advanced", type="secondary", use_container_width=True):
                    st.session_state.preprocessing_page = 2
                    st.rerun()
            
            if st.session_state.preprocessing_history:
                st.subheader("Preprocessing History")
                for i, step in enumerate(st.session_state.preprocessing_history):
                    st.write(f"{i+1}. {step}")

        if st.session_state.get('preprocessing_done', False) or st.session_state.get('advanced_preprocessing_done', False):
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Proceed to Dimensionality Reduction"):
                    st.session_state.step = 7
                    st.rerun()
            with col2:
                if st.button("Skip to Data Augmentation"):
                    st.session_state.skipped_steps.add(7)
                    st.session_state.step = 8
                    st.rerun()
                    
########################################################################################################################################################

    elif st.session_state.step == 7:
        st.header("Step 7: Dimensionality Reduction")

        X_train, X_test, y_train, y_test = get_current_data()
        if X_train is None:
            st.error("Please complete previous steps first")
            if st.button("Go back to Preprocessing"):
                st.session_state.step = 6
                st.rerun()
            st.stop()

        if ('advanced_preprocessing' in st.session_state.data_history and 
            (st.session_state.get('fft_applied', False) or st.session_state.get('opls_applied', False))):
            X_train, X_test, y_train, y_test = get_step_data('advanced_preprocessing')
            st.info("Using data from advanced preprocessing (FFT/OPLS applied in preprocessing step)")

        if ('dim_reducer' not in st.session_state or 
            st.button("Reset Dimensionality Reduction") or
            'current_X_train' not in st.session_state):
            st.session_state.dim_reducer = DimensionalityReduction(
                X_train=X_train, 
                X_test=X_test, 
                y_train=y_train, 
                y_test=y_test
            )
            st.session_state.current_X_train = X_train.copy()
            st.session_state.current_X_test = X_test.copy()
            st.session_state.dimensionality_done = False
            st.session_state.dimensionality_history = []
            if 'preprocessing_parameters' not in st.session_state:
                st.session_state.preprocessing_parameters = {}
            if 'dimensionality_steps' not in st.session_state.preprocessing_parameters:
                st.session_state.preprocessing_parameters['dimensionality_steps'] = []
            st.rerun()

        dim_reducer = st.session_state.dim_reducer
        current_X_train = st.session_state.current_X_train
        current_X_test = st.session_state.current_X_test

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Samples", current_X_train.shape[0])
        with col2:
            st.metric("Test Samples", current_X_test.shape[0])
        with col3:
            st.metric("Current Features", current_X_train.shape[1])
        original_features = X_train.shape[1]
        current_features = current_X_train.shape[1]
        reduction_pct = (1 - current_features/original_features) * 100 if original_features > 0 else 0
        st.metric("Feature Reduction", f"{reduction_pct:.1f}%")

        if st.session_state.dimensionality_history:
            st.subheader("Dimensionality Reduction History")
            for i, step in enumerate(st.session_state.dimensionality_history):
                st.write(f"{i+1}. {step}")

        st.subheader("Select Dimensionality Reduction Techniques")
        st.info("Techniques will be fitted on training data and applied to both training and test sets")

        with st.expander("Technique Descriptions"):
            st.markdown("""
            - **Scaling**: Standardizes features using standard, min-max, or robust scaling methods.
            - **PCA Analysis**: Reduces dimensionality by finding principal components that explain variance.
            - **Feature Selection**: Selects most important features using various statistical methods.
            
            Note: FFT Filtering and OPLS Analysis are now available in the Advanced Preprocessing section (Step 6, Page 2).
            """)

        selected_techniques = st.multiselect(
            "Choose dimensionality reduction techniques (applied in order):", 
            ['Scaling', 'PCA Analysis', 'Feature Selection'],
            key="selected_dim_techniques"
        )

        current_params = {}

        for idx, technique in enumerate(selected_techniques):
            st.markdown(f"### {technique} (Step {idx + 1})")

            if technique == 'Scaling':
                scaling_method = st.selectbox(
                    f"Select scaling method (Step {idx + 1}):",
                    ['standard', 'minmax', 'robust'],
                    key=f"scaling_method_{idx}"
                )
                current_params[f'scaling_{idx}'] = {'method': scaling_method}

            elif technique == 'PCA Analysis':
                pca_method = st.selectbox(
                    f"PCA selection method (Step {idx + 1}):",
                    ['variance', 'elbow', 'fixed'],
                    key=f"pca_method_{idx}"
                )
                pca_params = {}
                if pca_method == 'variance':
                    pca_params['variance_threshold'] = st.slider(
                        f"Variance Threshold (Step {idx + 1})", 
                        0.80, 0.99, 0.95, 0.01,
                        key=f"pca_variance_{idx}"
                    )
                elif pca_method == 'fixed':
                    max_components = min(50, current_X_train.shape[1])
                    pca_params['n_components'] = st.slider(
                        f"Number of Components (Step {idx + 1})", 
                        1, max_components, min(10, max_components),
                        key=f"pca_components_{idx}"
                    )
                col1, col2, col3 = st.columns(3)
                with col1:
                    pca_params['show_variance_plot'] = st.checkbox(f"Show Explained Variance (Step {idx + 1})", True, key=f"variance_plot_{idx}")
                with col2:
                    pca_params['show_2d_plot'] = st.checkbox(f"Show 2D Plot (Step {idx + 1})", True, key=f"2d_plot_{idx}")
                with col3:
                    pca_params['show_3d_plot'] = st.checkbox(f"Show 3D Plot (Step {idx + 1})", False, key=f"3d_plot_{idx}")
                current_params[f'pca_{idx}'] = {'method': pca_method,'parameters': pca_params}

            elif technique == 'Feature Selection':
                selection_method = st.selectbox(
                    f"Feature selection method (Step {idx + 1}):",
                    ['selectkbest', 'rfe', 'model_based', 'variance_threshold'],
                    key=f"selection_method_{idx}"
                )
                selection_params = {}
                if selection_method == 'selectkbest':
                    col1, col2 = st.columns(2)
                    with col1:
                        selection_params['k'] = st.slider(
                            f"Number of Features (k) (Step {idx + 1})", 
                            1, current_X_train.shape[1], 
                            min(10, current_X_train.shape[1]),
                            key=f"selectk_k_{idx}"
                        )
                    with col2:
                        selection_params['score_func_name'] = st.selectbox(
                            f"Score Function (Step {idx + 1}):", 
                            ['mutual_info_regression', 'f_regression'],
                            key=f"score_func_{idx}"
                        )
                elif selection_method == 'rfe':
                    col1, col2 = st.columns(2)
                    with col1:
                        selection_params['n_features'] = st.slider(
                            f"Number of Features (Step {idx + 1})", 
                            1, current_X_train.shape[1],
                            min(10, current_X_train.shape[1]),
                            key=f"rfe_n_{idx}"
                        )
                    with col2:
                        selection_params['estimator_type'] = st.selectbox(
                            f"Estimator (Step {idx + 1}):", 
                            ['RandomForest', 'LinearRegression'],
                            key=f"rfe_estimator_{idx}"
                        )
                elif selection_method == 'model_based':
                    col1, col2 = st.columns(2)
                    with col1:
                        selection_params['threshold'] = st.selectbox(
                            f"Threshold (Step {idx + 1}):", 
                            ['mean', 'median'],
                            key=f"model_threshold_{idx}"
                        )
                    with col2:
                        selection_params['estimator_type'] = st.selectbox(
                            f"Estimator (Step {idx + 1}):", 
                            ['RandomForest', 'LinearRegression'],
                            key=f"model_estimator_{idx}"
                        )
                elif selection_method == 'variance_threshold':
                    selection_params['threshold'] = st.slider(
                        f"Variance Threshold (Step {idx + 1})", 
                        0.0, 1.0, 0.01, 0.01,
                        key=f"var_threshold_{idx}"
                    )
                current_params[f'feature_selection_{idx}'] = {'method': selection_method,'parameters': selection_params}

        if selected_techniques and st.button("Apply Dimensionality Reduction"):
            try:
                current_reducer = DimensionalityReduction(
                    X_train=current_X_train, 
                    X_test=current_X_test, 
                    y_train=y_train, 
                    y_test=y_test
                )
                applied_params = {}
                applied_steps = []
                all_successful = True

                for idx, technique in enumerate(selected_techniques):
                    st.write(f"Applying `{technique}` (Step {idx + 1})...")
                    st.write(f"  Current training data shape: {current_reducer.X_train.shape}")
                    st.write(f"  Current test data shape: {current_reducer.X_test.shape}")
                    try:
                        if technique == 'Scaling':
                            params = current_params[f'scaling_{idx}']
                            current_reducer.apply_scaling(scaling_method=params['method'])
                            applied_params[f'scaling_{idx}'] = params
                            applied_steps.append(f"Scaling: {params['method']}")
                            st.success(f"Applied {params['method']} scaling - fitted on training data")

                        elif technique == 'PCA Analysis':
                            params = current_params[f'pca_{idx}']
                            pca_method = params['method']
                            pca_params = params['parameters']
                            if current_reducer.X_train_scaled is None:
                                current_reducer.apply_scaling(scaling_method="standard")
                            if pca_method == 'variance':
                                X_train_reduced, X_test_reduced, n_components = current_reducer.pca_analysis(
                                    method='variance',
                                    variance_threshold=pca_params['variance_threshold'],
                                    use_scaled=True
                                )
                            elif pca_method == 'elbow':
                                X_train_reduced, X_test_reduced, n_components = current_reducer.pca_analysis(
                                    method='elbow',
                                    use_scaled=True
                                )
                            else:
                                X_train_reduced, X_test_reduced, n_components = current_reducer.pca_analysis(
                                    method='fixed',
                                    n_components=pca_params['n_components'],
                                    use_scaled=True
                                )
                            applied_params[f'pca_{idx}'] = params
                            applied_steps.append(f"PCA ({pca_method}): {n_components} components")
                            st.success(f"PCA completed: Reduced to {n_components} components - fitted on training data")
                            if pca_params.get('show_variance_plot', False):
                                st.subheader(f"Explained Variance Analysis (Step {idx + 1})")
                                current_reducer.plot_explained_variance()
                            if pca_params.get('show_2d_plot', False) and n_components >= 2:
                                st.subheader(f"2D PCA Visualization (Step {idx + 1})")
                                current_reducer.plot_scores_pairgrid(use_train=True, pcs=(1,2), kde=True, ellipses=True, title=f"2D PCA Projection - Training Data (Step {idx + 1})")
                            if pca_params.get('show_3d_plot', False) and n_components >= 3:
                                st.subheader(f"3D PCA Visualization (Step {idx + 1})")
                                current_reducer.plot_scores_pairgrid(use_train=True, pcs=(1,2,3), kde=False, ellipses=False, title=f"3D PCA Projection - Training Data (Step {idx + 1})")
                            current_reducer.X_train = X_train_reduced
                            current_reducer.X_test = X_test_reduced

                        elif technique == 'Feature Selection':
                            params = current_params[f'feature_selection_{idx}']
                            selection_method = params['method']
                            selection_params = params['parameters']
                            kwargs = {}
                            original_features = current_reducer.X_train.shape[1]
                            if selection_method == 'selectkbest':
                                kwargs['k'] = selection_params['k']
                                if selection_params['score_func_name'] == 'mutual_info_regression':
                                    from sklearn.feature_selection import mutual_info_regression
                                    kwargs['score_func'] = mutual_info_regression
                                else:
                                    from sklearn.feature_selection import f_regression
                                    kwargs['score_func'] = f_regression
                            elif selection_method == 'rfe':
                                kwargs['n_features'] = selection_params['n_features']
                                if selection_params['estimator_type'] == 'RandomForest':
                                    from sklearn.ensemble import RandomForestRegressor
                                    kwargs['estimator'] = RandomForestRegressor(n_estimators=50, random_state=42)
                                else:
                                    from sklearn.linear_model import LinearRegression
                                    kwargs['estimator'] = LinearRegression()
                            elif selection_method == 'model_based':
                                kwargs['threshold'] = selection_params['threshold']
                                if selection_params['estimator_type'] == 'RandomForest':
                                    from sklearn.ensemble import RandomForestRegressor
                                    kwargs['estimator'] = RandomForestRegressor(n_estimators=50, random_state=42)
                                else:
                                    from sklearn.linear_model import LinearRegression
                                    kwargs['estimator'] = LinearRegression()
                            elif selection_method == 'variance_threshold':
                                kwargs['threshold'] = selection_params['threshold']
                            use_scaled = selection_method != 'variance_threshold'
                            if use_scaled and current_reducer.X_train_scaled is None:
                                current_reducer.apply_scaling(scaling_method="standard")
                            X_train_reduced, X_test_reduced = current_reducer.feature_selection(
                                method=selection_method, 
                                use_scaled=use_scaled,
                                **kwargs
                            )
                            applied_params[f'feature_selection_{idx}'] = params
                            applied_steps.append(f"Feature Selection ({selection_method}): {original_features} → {X_train_reduced.shape[1]} features")
                            st.success(f"Feature Selection completed: Reduced from {original_features} to {X_train_reduced.shape[1]} features - fitted on training data")
                            current_reducer.X_train = X_train_reduced
                            current_reducer.X_test = X_test_reduced

                        st.info(f"Training data shape after {technique}: {current_reducer.X_train.shape}")
                        st.info(f"Test data shape after {technique}: {current_reducer.X_test.shape}")
                    except Exception as technique_error:
                        st.error(f"Error in {technique}: {str(technique_error)}")
                        all_successful = False
                        break

                if all_successful and len(applied_steps) == len(selected_techniques):
                    final_X_train = current_reducer.get_transformed_data(use_train=True)
                    final_X_test = current_reducer.get_transformed_data(use_train=False)
                    if final_X_train is None:
                        final_X_train = current_reducer.X_train
                        final_X_test = current_reducer.X_test
                    st.session_state.current_X_train = final_X_train
                    st.session_state.current_X_test = final_X_test
                    st.session_state.X_train = final_X_train  
                    st.session_state.X_test = final_X_test
                    st.session_state.dim_reducer = current_reducer
                    st.session_state.dimensionality_history.extend(applied_steps)
                    st.session_state.preprocessing_parameters['dimensionality_steps'].extend(applied_steps)
                    if applied_params:
                        st.session_state.preprocessing_parameters['dimensionality_parameters'] = applied_params
                    save_step_data('dimensionality', final_X_train, final_X_test, y_train, y_test)
                    st.session_state.dimensionality_done = True
                    st.success("All dimensionality reduction techniques applied successfully!")
                    st.success("Fitted on training data and transformed both training and test sets")
                else:
                    st.warning("Some techniques failed. Please try again.")
            except Exception as e:
                st.error(f"Error applying dimensionality reduction: {str(e)}")

        if not selected_techniques:
            st.session_state.dimensionality_done = True
            st.info("No dimensionality reduction techniques selected.")

        if st.session_state.dimensionality_done:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Proceed to Data Augmentation"):
                    st.session_state.step = 8
                    st.rerun()
            with col2:
                if st.button("Reset All Changes"):
                    original_X_train, original_X_test, original_y_train, original_y_test = get_current_data()
                    if original_X_train is None:
                        if hasattr(st.session_state, 'X_train') and hasattr(st.session_state, 'X_test'):
                            original_X_train = st.session_state.X_train.copy()
                            original_X_test = st.session_state.X_test.copy()
                        else:
                            st.error("No data available to reset to")
                            st.stop()
                    st.session_state.current_X_train = original_X_train.copy()
                    st.session_state.current_X_test = original_X_test.copy()
                    st.session_state.X_train = original_X_train.copy()
                    st.session_state.X_test = original_X_test.copy()
                    st.session_state.dim_reducer = DimensionalityReduction(
                        X_train=original_X_train, 
                        X_test=original_X_test, 
                        y_train=y_train, 
                        y_test=y_test
                    )
                    st.session_state.dimensionality_history = []
                    st.session_state.dimensionality_done = False
                    if 'preprocessing_parameters' not in st.session_state:
                        st.session_state.preprocessing_parameters = {}
                    st.session_state.preprocessing_parameters['dimensionality_steps'] = []
                    if 'dimensionality_parameters' in st.session_state.preprocessing_parameters:
                        del st.session_state.preprocessing_parameters['dimensionality_parameters']
                    st.success("All dimensionality reduction changes reset")
                    st.rerun()

#########################################################################################################################################################

    elif st.session_state.step == 8:
        st.header("Step 8: Data Augmentation")
        
        X_train, X_test, y_train, y_test = get_current_data()
        if X_train is None:
            st.error("No data available. Please complete previous steps.")
            return
        
        if 'augmentation_history' not in st.session_state:
            st.session_state.augmentation_history = []
        if 'augmented_X_train' not in st.session_state:
            st.session_state.augmented_X_train = X_train.copy()
        if 'augmented_y_train' not in st.session_state:
            st.session_state.augmented_y_train = y_train.copy()
        if 'augmentation_done' not in st.session_state:
            st.session_state.augmentation_done = False
        if 'augmentation' not in st.session_state.preprocessing_parameters:
            st.session_state.preprocessing_parameters['augmentation'] = []

        X_train_current = st.session_state.augmented_X_train
        y_train_current = st.session_state.augmented_y_train
        
        st.write(f"Current training dataset shape: {X_train_current.shape}")
        st.info(f"Test dataset unchanged: {X_test.shape}")
        
        if st.session_state.augmentation_history:
            st.subheader("Augmentation History")
            for i, aug in enumerate(st.session_state.augmentation_history):
                st.write(f"{i+1}. {aug}")
        
        st.subheader("Apply Multiple Augmentations (Training Data Only)")
        with st.expander("Augmentation Technique Descriptions"):
            st.markdown("""
            - **Add Spectra:** Combines two or more spectra to create new synthetic samples with blended characteristics.  
            - **Mixup:** Generates new samples by interpolating features and labels between pairs of existing samples.  
            - **Spectral Shift:** Shifts spectral features left or right to simulate peak displacement or instrument variation.  
            - **Gaussian Noise:** Adds small random noise to spectra, mimicking measurement variability or sensor noise.
            """)

        selected_augmentations = st.multiselect(
            "Select augmentation techniques to apply:", 
            ['Add Spectra', 'Mixup', 'Spectral Shift', 'Gaussian Noise']
        )
        
        aug_params = {}
        for aug_method in selected_augmentations:
            st.markdown(f"### Parameters for {aug_method}")
            
            if aug_method == 'Mixup':
                col1, col2 = st.columns(2)
                with col1:
                    num_copies = st.number_input(f"Number of synthetic samples ({aug_method})", 1, 2000, 200, key=f"num_{aug_method}")
                with col2:
                    alpha = st.slider(f"Mixup Alpha ({aug_method})", 0.1, 1.0, 0.4, 0.1, key=f"alpha_{aug_method}")
                aug_params[aug_method] = {
                    'num_copies': num_copies,
                    'alpha': alpha
                }
            
            elif aug_method == 'Gaussian Noise':
                col1, col2 = st.columns(2)
                with col1:
                    aug_params[aug_method] = {
                        'num_copies': st.number_input(f"Augmenting your dataset by 2–5× ({aug_method})", 1, 50, 2, key=f"num_{aug_method}")
                    }
                with col2:
                    aug_params[aug_method]['std'] = st.number_input(f"Noise standard deviation ({aug_method})", 0.001, 1.0, 0.01, 0.001, key=f"std_{aug_method}")
            
            elif aug_method == 'Add Spectra':
                st.info("Add Spectra combines existing spectra. Parameters may vary based on implementation.")
                aug_params[aug_method] = {
                    'num_copies': st.number_input(f"Augmenting your dataset by 2–5× ({aug_method})", 1, 100, 20, key=f"num_{aug_method}")
                }
            
            elif aug_method == 'Spectral Shift':
                st.info("Spectral Shift applies wavelength shifts to spectra.")
                col1, col2 = st.columns(2)
                with col1:
                    aug_params[aug_method] = {
                        'num_copies': st.number_input(f"Augmenting your dataset by 2–5×({aug_method})", 1, 100, 20, key=f"num_{aug_method}")
                    }
                with col2:
                    aug_params[aug_method]['shift_range'] = st.slider(f"Shift range ({aug_method})", 1, 20, 5, key=f"shift_{aug_method}")
        
        if selected_augmentations and st.button("Apply Selected Augmentations"):
            try:
                augmentation_successful = True
                
                for aug_method in selected_augmentations:
                    st.write(f"Applying {aug_method} to training data...")
                    
                    def is_empty_data(data):
                        if hasattr(data, 'empty'):
                            return data.empty
                        else:
                            return len(data) == 0 or data.size == 0
                    
                    if is_empty_data(X_train_current) or is_empty_data(y_train_current):
                        st.error(f"Empty data detected before {aug_method}")
                        augmentation_successful = False
                        break
                    
                    if X_train_current.shape[0] != y_train_current.shape[0]:
                        st.error(f"Data inconsistency before {aug_method}! X: {X_train_current.shape[0]}, y: {y_train_current.shape[0]}")
                        augmentation_successful = False
                        break
                    
                    st.write(f"  Current training data shape: X={X_train_current.shape}, y={y_train_current.shape}")
                    
                    try:
                        augmentor = DataAugmentor(X_train_current.copy(), y_train_current.copy())
                    except Exception as aug_init_error:
                        st.error(f"Failed to initialize augmentor for {aug_method}: {str(aug_init_error)}")
                        augmentation_successful = False
                        break
                    
                    try:
                        if aug_method == 'Mixup':
                            params = aug_params[aug_method]
                            if params['num_copies'] <= 0:
                                st.warning(f"Invalid num_copies for {aug_method}, skipping...")
                                continue
                            
                            X_new, y_new = augmentor.mixup(
                                num_copies=int(params['num_copies']), 
                                alpha=float(params['alpha'])
                            )
                            history_msg = f"{aug_method}: {params['num_copies']} samples added"
                            
                        elif aug_method == 'Gaussian Noise':
                            params = aug_params[aug_method]
                            if params['num_copies'] <= 0 or params['std'] <= 0:
                                st.warning(f"Invalid parameters for {aug_method}, skipping...")
                                continue
                            
                            X_new, y_new = augmentor.gaussian_noise(
                                num_copies=int(params['num_copies']), 
                                std=float(params['std'])
                            )
                            history_msg = f"{aug_method}: {params['num_copies']} samples added (std={params['std']})"           

                        elif aug_method == 'Add Spectra':
                            params = aug_params[aug_method]
                            try:
                                if params['num_copies'] <= 0:
                                    st.warning(f"Invalid num_copies for {aug_method}, using default...")
                                    X_new, y_new = augmentor.add_spectra()
                                    history_msg = f"{aug_method}: Applied with default parameters"
                                else:
                                    X_new, y_new = augmentor.add_spectra(num_copies=int(params['num_copies']))
                                    history_msg = f"{aug_method}: {params['num_copies']} samples requested"
                            except Exception as add_error:
                                st.error(f"Error in {aug_method}: {str(add_error)}")
                                augmentation_successful = False
                                break
                        
                        elif aug_method == 'Spectral Shift':
                            params = aug_params[aug_method]
                            try:
                                if params['num_copies'] <= 0:
                                    st.warning(f"Invalid num_copies for {aug_method}, skipping...")
                                    continue
                                
                                max_features = X_train_current.shape[1]
                                shift_range = min(int(params['shift_range']), max_features // 10)
                                
                                if shift_range <= 0:
                                    shift_range = 1
                                
                                X_new, y_new = augmentor.spectral_shift(
                                    num_copies=int(params['num_copies']),
                                    shift_range=shift_range
                                )
                                history_msg = f"{aug_method}: {params['num_copies']} samples added (shift={shift_range})"
                                
                            except Exception as shift_error:
                                st.error(f"Error in {aug_method}: {str(shift_error)}")
                                augmentation_successful = False
                                break
                        
                        else:
                            st.warning(f"Unknown augmentation method: {aug_method}")
                            continue
                        
                        if X_new is None or y_new is None:
                            st.error(f"{aug_method} returned None values, skipping...")
                            augmentation_successful = False
                            break
                        
                        if len(X_new) == 0 or len(y_new) == 0:
                            st.error(f"{aug_method} returned empty data, skipping...")
                            augmentation_successful = False
                            break
                        
                        if X_new.shape[0] != y_new.shape[0]:
                            st.error(f"Output inconsistency from {aug_method}! X: {X_new.shape[0]}, y: {y_new.shape[0]}")
                            augmentation_successful = False
                            break
                        
                        if X_new.shape[1] != X_train_current.shape[1]:
                            st.warning(f"{aug_method} changed number of features from {X_train_current.shape[1]} to {X_new.shape[1]}")
                        
                        X_train_current = X_new.copy()
                        y_train_current = y_new.copy()
                        
                        st.session_state.preprocessing_parameters['augmentation'].append({
                            'method': aug_method,
                            'parameters': aug_params[aug_method],
                            'result_shape': list(X_new.shape)
                        })
                        
                        st.session_state.augmented_X_train = X_train_current.copy()
                        st.session_state.augmented_y_train = y_train_current.copy()
                        
                        st.session_state.augmentation_history.append(history_msg)
                        
                        st.success(f"  {aug_method} completed: {X_train_current.shape[0]} total training samples")
                        
                    except IndexError as idx_error:
                        st.error(f"Index error in {aug_method}: {str(idx_error)}")
                        augmentation_successful = False
                        break
                    except ValueError as val_error:
                        st.error(f"Value error in {aug_method}: {str(val_error)}")
                        augmentation_successful = False
                        break
                    except Exception as method_error:
                        st.error(f"Unexpected error in {aug_method}: {str(method_error)}")
                        augmentation_successful = False
                        break
                
                if augmentation_successful and X_train_current.shape[0] == y_train_current.shape[0] and len(X_train_current) > 0:
                    st.success(f"All augmentations completed successfully!")
                    st.info(f"Training size: {X_train.shape[0]} → {X_train_current.shape[0]} (added {X_train_current.shape[0] - X_train.shape[0]} samples)")
                    st.info(f"Test size unchanged: {X_test.shape[0]}")
                    
                    try:
                        save_step_data('augmentation', 
                                     st.session_state.augmented_X_train, 
                                     X_test,
                                     st.session_state.augmented_y_train, 
                                     y_test)
                        st.session_state.augmentation_done = True
                    except Exception as save_error:
                        st.warning(f"Data saved with warning: {str(save_error)}")
                        st.session_state.augmentation_done = True
                    
                else:
                    st.error("Augmentation failed - please try again or skip this step")
                    st.session_state.augmentation_done = False
                
            except Exception as e:
                st.error(f"Error in augmentation: {str(e)}")
                st.session_state.augmentation_done = False
        
        if 'augmented_X_train' in st.session_state and len(st.session_state.augmented_X_train) > 0:
            try:
                import matplotlib.pyplot as plt 
                fig, ax = plt.subplots(figsize=(12, 6))
                sample_size = min(100, st.session_state.augmented_X_train.shape[0])
                for i in range(sample_size):
                    if hasattr(st.session_state.augmented_X_train, 'iloc'):
                        ax.plot(st.session_state.augmented_X_train.iloc[i], alpha=0.4)
                    else:
                        ax.plot(st.session_state.augmented_X_train[i], alpha=0.4)
                ax.set_title(f"Current Augmented Training Spectra (showing {sample_size} samples)")
                ax.set_xlabel("Features")
                ax.set_ylabel("Intensity")
                st.pyplot(fig)
                plt.close(fig)
            except Exception as plot_error:
                st.warning(f"Could not plot augmented spectra: {plot_error}")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Reset All Augmentations"):
                X_train, X_test, y_train, y_test = get_current_data()
                if X_train is None:
                    st.error("No original data available to reset to")
                    return
                
                st.session_state.augmented_X_train = X_train.copy()
                st.session_state.augmented_y_train = y_train.copy()
                st.session_state.augmentation_history = []
                st.session_state.augmentation_done = False
                
                if 'preprocessing_parameters' in st.session_state:
                    st.session_state.preprocessing_parameters['augmentation'] = []
                
                st.success("All augmentations reset")
                st.rerun()
                    
        with col2:
            if not st.session_state.augmentation_done and st.button("Finalize Augmentation"):
                try:
                    save_step_data('augmentation', 
                                st.session_state.augmented_X_train, 
                                X_test,
                                st.session_state.augmented_y_train, 
                                y_test)
                    st.session_state.augmentation_done = True
                    st.success("Augmentation finalized successfully!")
                    st.rerun()
                except Exception as save_error:
                    st.error(f"Error finalizing augmentation: {str(save_error)}")

        with col3:
            if not st.session_state.augmentation_done and st.button("Skip Augmentation"):
                try:
                    X_train, X_test, y_train, y_test = get_current_data()
                    save_step_data('augmentation', X_train, X_test, y_train, y_test)
                    st.session_state.augmentation_done = True
                    st.session_state.augmented_X_train = X_train.copy()
                    st.session_state.augmented_y_train = y_train.copy()
                    st.success("Skipped augmentation")
                    st.rerun()
                except Exception as skip_error:
                    st.error(f"Error skipping augmentation: {str(skip_error)}")

        if st.session_state.augmentation_done:
            st.divider()
            st.success("Augmentation step completed - ready to proceed")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Proceed to MPLS SPC (Step 9)", type="primary"):
                    st.session_state.step = 9
                    st.rerun()
            with col_b:
                if st.button("Skip MPLS and go to Model Training (Step 10)"):
                    st.session_state.step = 10
                    st.rerun()
        else:
            st.info("Apply augmentations or click 'Skip Augmentation' or 'Finalize Augmentation' to proceed")


   #######################################################################################################################################################
    elif st.session_state.step == 9:
        st.header("Step 9: MPLS - Multiway PLS for Batch Process Monitoring")
        
        X_train, X_test, y_train, y_test = get_current_data()
        if X_train is None:
            st.error("Please complete previous steps first")
            if st.button("Go back to Data Augmentation"):
                st.session_state.step = 8
                st.rerun()
            st.stop()
        
        st.info("MPLS (Multiway PLS) is used for batch process monitoring with Statistical Process Control (SPC)")
        
        if 'mpls_model' not in st.session_state or st.button("Reset MPLS Analysis"):
            st.session_state.mpls_model = None
            st.session_state.mpls_done = False
            st.session_state.mpls_parameters = {}
            st.rerun()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Samples", X_train.shape[0])
        with col2:
            st.metric("Test Samples", X_test.shape[0])
        with col3:
            st.metric("Features", X_train.shape[1])
        
        with st.expander("MPLS Method Description"):
            st.markdown("""
            **Multiway Partial Least Squares (MPLS)** analyzes batch processes with:
            - **Batch Evolution Model (BEM)**: Monitors process progression over time
            - **DModX**: Distance to Model in X-space (detects unusual patterns)
            - **Hotelling's T²**: Multivariate control statistic
            - **SPC Charts**: Statistical Process Control with ±3σ limits
            
            This method is ideal for monitoring manufacturing batches, chemical processes, and time-series quality control.
            """)
        
        st.subheader("MPLS Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            batch_structure = st.selectbox(
                "Batch Data Structure:",
                ['Auto-create Single Batch', 'Multiple Batches with IDs', 'Manual Batch Configuration']
            )
        
        with col2:
            scaling_method = st.selectbox(
                "Scaling Method:",
                ['standard', 'minmax', 'robust', 'none']
            )
        
        batch_ids_train = None
        batch_ids_test = None
        time_col = None
        batch_col = None
        n_batches = 1
        n_timepoints = X_train.shape[0]
        
        X_train_copy = X_train.copy()
        X_test_copy = X_test.copy() if X_test is not None else None
        
        if batch_structure == 'Multiple Batches with IDs':
            if isinstance(X_train, pd.DataFrame):
                available_cols = list(X_train.columns)
                
                col1, col2 = st.columns(2)
                with col1:
                    if 'Batch_ID' in available_cols:
                        batch_col = st.selectbox("Batch ID Column:", available_cols, index=available_cols.index('Batch_ID'))
                    else:
                        batch_col = st.selectbox("Batch ID Column:", available_cols)
                    
                    if batch_col:
                        batch_ids_train = X_train[batch_col].values
                        if X_test is not None and isinstance(X_test, pd.DataFrame) and batch_col in X_test.columns:
                            batch_ids_test = X_test[batch_col].values
                        
                        st.info(f"Found {len(np.unique(batch_ids_train))} unique batches in training data")
                        
                        X_train_copy = X_train.drop(columns=[batch_col])
                        if X_test is not None and isinstance(X_test, pd.DataFrame) and batch_col in X_test.columns:
                            X_test_copy = X_test.drop(columns=[batch_col])
                
                with col2:
                    time_cols = [col for col in available_cols if 'time' in col.lower()]
                    if time_cols:
                        time_col = st.selectbox("Time Column (optional):", ['None'] + time_cols)
                        time_col = None if time_col == 'None' else time_col
                        
                        if time_col:
                            X_train_copy = X_train_copy.drop(columns=[time_col])
                            if X_test_copy is not None and isinstance(X_test_copy, pd.DataFrame) and time_col in X_test_copy.columns:
                                X_test_copy = X_test_copy.drop(columns=[time_col])
            else:
                st.error("Multiple Batches with IDs requires DataFrame input. Please select a different option.")
                batch_structure = 'Auto-create Single Batch'
        
        elif batch_structure == 'Manual Batch Configuration':
            col1, col2 = st.columns(2)
            with col1:
                max_batches = X_train.shape[0]
                n_batches = st.number_input(
                    "Number of Batches:", 
                    min_value=1, 
                    max_value=max_batches, 
                    value=min(10, max_batches),
                    help="Total samples will be divided into this many batches"
                )
            with col2:
                n_timepoints = X_train.shape[0] // n_batches
                st.metric("Timepoints per Batch", n_timepoints)
            
            actual_samples = n_batches * n_timepoints
            
            if X_train.shape[0] != actual_samples:
                st.warning(f"Using {actual_samples} samples out of {X_train.shape[0]} (truncating last {X_train.shape[0] - actual_samples} samples)")
            
            batch_ids_train = np.repeat(np.arange(n_batches), n_timepoints)
            
            if isinstance(X_train_copy, pd.DataFrame):
                X_train_copy = X_train_copy.iloc[:actual_samples].reset_index(drop=True)
            else:
                X_train_copy = X_train_copy[:actual_samples]
            
            if y_train is not None:
                if isinstance(y_train, pd.Series) or isinstance(y_train, pd.DataFrame):
                    y_train = y_train.iloc[:actual_samples].reset_index(drop=True)
                else:
                    y_train = y_train[:actual_samples]
            
            if X_test is not None:
                n_batches_test = X_test.shape[0] // n_timepoints
                if n_batches_test < 1:
                    n_batches_test = 1
                    n_timepoints_test = X_test.shape[0]
                else:
                    n_timepoints_test = n_timepoints
                
                actual_samples_test = n_batches_test * n_timepoints_test
                batch_ids_test = np.repeat(np.arange(n_batches_test), n_timepoints_test)
                
                if isinstance(X_test_copy, pd.DataFrame):
                    X_test_copy = X_test_copy.iloc[:actual_samples_test].reset_index(drop=True)
                else:
                    X_test_copy = X_test_copy[:actual_samples_test]
                
                if y_test is not None:
                    if isinstance(y_test, pd.Series) or isinstance(y_test, pd.DataFrame):
                        y_test = y_test.iloc[:actual_samples_test].reset_index(drop=True)
                    else:
                        y_test = y_test[:actual_samples_test]
        
        else:
            st.info("Using all training data as a single batch")
            n_timepoints = st.slider(
                "Timepoints per Batch:",
                min_value=10,
                max_value=X_train.shape[0],
                value=min(50, X_train.shape[0]),
                help="Each batch will have this many timepoints"
            )
            
            n_batches = X_train.shape[0] // n_timepoints
            
            if n_batches < 1:
                n_batches = 1
                n_timepoints = X_train.shape[0]
                st.warning(f"Adjusted to use all {n_timepoints} samples as 1 batch")
            
            actual_samples = n_batches * n_timepoints
            
            st.info(f"Creating {n_batches} batch(es) with {n_timepoints} timepoints each")
            
            if X_train.shape[0] != actual_samples:
                st.warning(f"Using {actual_samples} samples out of {X_train.shape[0]} (truncating last {X_train.shape[0] - actual_samples} samples)")
            import numpy as np 
            batch_ids_train = np.repeat(np.arange(n_batches), n_timepoints)
            
            if isinstance(X_train_copy, pd.DataFrame):
                X_train_copy = X_train_copy.iloc[:actual_samples].reset_index(drop=True)
            else:
                X_train_copy = X_train_copy[:actual_samples]
            
            if y_train is not None:
                if isinstance(y_train, pd.Series) or isinstance(y_train, pd.DataFrame):
                    y_train = y_train.iloc[:actual_samples].reset_index(drop=True)
                else:
                    y_train = y_train[:actual_samples]
            
            if X_test is not None:
                n_batches_test = X_test.shape[0] // n_timepoints
                if n_batches_test < 1:
                    n_batches_test = 1
                    n_timepoints_test = X_test.shape[0]
                else:
                    n_timepoints_test = n_timepoints
                
                actual_samples_test = n_batches_test * n_timepoints_test
                batch_ids_test = np.repeat(np.arange(n_batches_test), n_timepoints_test)
                
                if isinstance(X_test_copy, pd.DataFrame):
                    X_test_copy = X_test_copy.iloc[:actual_samples_test].reset_index(drop=True)
                else:
                    X_test_copy = X_test_copy[:actual_samples_test]
                
                if y_test is not None:
                    if isinstance(y_test, pd.Series) or isinstance(y_test, pd.DataFrame):
                        y_test = y_test.iloc[:actual_samples_test].reset_index(drop=True)
                    else:
                        y_test = y_test[:actual_samples_test]
        
        st.info(f"Final data shapes - X_train: {X_train_copy.shape}, batch_ids: {len(batch_ids_train) if batch_ids_train is not None else 'None'}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            n_components = st.slider("Number of PLS Components:", 1, 10, 2)
        with col2:
            use_bem = st.checkbox("Use Batch Evolution Model (BEM)", value=True, 
                                help="BEM uses time index as dummy Y variable")
        with col3:
            show_loadings = st.checkbox("Show Loadings Plot", value=True)
        
        st.subheader("Visualization Options")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            plot_dmodx = st.checkbox("DModX SPC Chart", value=True)
        with col2:
            plot_t2 = st.checkbox("Hotelling's T² Chart", value=True)
        with col3:
            plot_scores = st.checkbox("Scores Scatter", value=True)
        with col4:
            plot_variance = st.checkbox("Explained Variance", value=False)
        
        if st.button("Fit MPLS Model", type="primary"):
            try:
                with st.spinner("Fitting MPLS model..."):
                    
                    st.write(f"Debug - X_train_copy shape: {X_train_copy.shape}")
                    st.write(f"Debug - batch_ids_train length: {len(batch_ids_train) if batch_ids_train is not None else 'None'}")
                    st.write(f"Debug - y_train shape: {y_train.shape if y_train is not None else 'None'}")
                    
                    mpls = MultiWayPLS(
                        X_train=X_train_copy,
                        X_test=X_test_copy,
                        y_train=y_train,
                        y_test=y_test,
                        batch_ids_train=batch_ids_train,
                        batch_ids_test=batch_ids_test,
                        time_col=None
                    )
                    
                    mpls.apply_scaling(scaling_method=scaling_method)
                    st.success(f"Applied {scaling_method} scaling")
                    
                    mpls.fit_mpls(n_components=n_components, use_scaled=True, use_bem=use_bem)
                    
                    st.session_state.mpls_model = mpls
                    st.session_state.mpls_parameters = {
                        'n_components': n_components,
                        'scaling_method': scaling_method,
                        'use_bem': use_bem,
                        'batch_structure': batch_structure,
                        'n_batches_train': mpls.n_batches_train,
                        'n_timepoints': mpls.n_timepoints,
                        'n_features': mpls.n_features
                    }
                    
                    if 'preprocessing_parameters' not in st.session_state:
                        st.session_state.preprocessing_parameters = {}
                    st.session_state.preprocessing_parameters['mpls'] = st.session_state.mpls_parameters
                    
                    st.success(f"MPLS model fitted with {n_components} components!")
                    st.info(f"Batches: {mpls.n_batches_train} | Timepoints: {mpls.n_timepoints} | Features: {mpls.n_features}")
                    
            except Exception as e:
                st.error(f"Error fitting MPLS model: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        
        if st.session_state.mpls_model is not None:
            mpls = st.session_state.mpls_model
            
            st.divider()
            st.subheader("MPLS Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("PLS Components", mpls.pls_model.n_components)
            with col2:
                st.metric("Training Batches", mpls.n_batches_train)
            with col3:
                st.metric("Timepoints per Batch", mpls.n_timepoints)
            
            if show_loadings:
                st.subheader("PLS Loadings")
                try:
                    components_to_plot = list(range(1, min(mpls.pls_model.n_components + 1, 4)))
                    mpls.plot_loadings(components=components_to_plot, title="PLS Loadings")
                except Exception as e:
                    st.warning(f"Could not plot loadings: {str(e)}")
            
            if plot_variance:
                st.subheader("Explained Variance")
                try:
                    mpls.plot_explained_variance()
                except Exception as e:
                    st.warning(f"Could not plot variance: {str(e)}")
            
            st.divider()
            st.subheader("Statistical Process Control (SPC) - Training Batches")
            
            batch_labels_train = [f"Batch {i+1}" for i in range(mpls.n_batches_train)]
            
            if plot_dmodx:
                st.markdown("### DModX - Distance to Model")
                try:
                    mpls.plot_spc_chart(
                        metric='DModX',
                        use_train=True,
                        batch_labels=batch_labels_train,
                        title="DModX - SPC Chart (Training Batches)",
                        highlight_ooc=True
                    )
                except Exception as e:
                    st.error(f"Error plotting DModX: {str(e)}")
            
            if plot_t2:
                st.markdown("### Hotelling's T² Statistic")
                try:
                    mpls.plot_spc_chart(
                        metric='T2',
                        use_train=True,
                        batch_labels=batch_labels_train,
                        title="Hotelling's T² - SPC Chart (Training Batches)",
                        highlight_ooc=True
                    )
                except Exception as e:
                    st.error(f"Error plotting T²: {str(e)}")
            
            st.markdown("### Score t[1] - First Component")
            try:
                mpls.plot_spc_chart(
                    metric='score',
                    use_train=True,
                    batch_labels=batch_labels_train,
                    title="Score t[1] - SPC Chart (Training Batches)",
                    highlight_ooc=True
                )
            except Exception as e:
                st.error(f"Error plotting scores: {str(e)}")
            
            if plot_scores and mpls.pls_model.n_components >= 2:
                st.markdown("### PLS Scores Scatter Plot")
                try:
                    mpls.plot_scores_scatter(
                        components=(1, 2),
                        use_train=True,
                        batch_labels=batch_labels_train
                    )
                except Exception as e:
                    st.warning(f"Could not plot scores scatter: {str(e)}")
            
            if mpls.X_test is not None and hasattr(mpls, 'n_batches_test') and mpls.n_batches_test > 0:
                st.divider()
                st.subheader("SPC - Test Batches")
                
                batch_labels_test = [f"Test Batch {i+1}" for i in range(mpls.n_batches_test)]
                
                col1, col2 = st.columns(2)
                with col1:
                    show_test_dmodx = st.checkbox("Show Test DModX", value=True)
                with col2:
                    show_test_t2 = st.checkbox("Show Test T²", value=True)
                
                if show_test_dmodx:
                    st.markdown("### Test DModX")
                    try:
                        mpls.plot_spc_chart(
                            metric='DModX',
                            use_train=False,
                            batch_labels=batch_labels_test,
                            title="DModX - SPC Chart (Test Batches)",
                            highlight_ooc=True
                        )
                    except Exception as e:
                        st.error(f"Error plotting test DModX: {str(e)}")
                
                if show_test_t2:
                    st.markdown("### Test T²")
                    try:
                        mpls.plot_spc_chart(
                            metric='T2',
                            use_train=False,
                            batch_labels=batch_labels_test,
                            title="Hotelling's T² - SPC Chart (Test Batches)",
                            highlight_ooc=True
                        )
                    except Exception as e:
                        st.error(f"Error plotting test T²: {str(e)}")
            
            st.divider()
            st.subheader("Outlier Detection")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Detect Training Outliers"):
                    try:
                        outliers = mpls.detect_outliers(use_train=True)
                        if outliers is not None and len(outliers) > 0:
                            st.warning(f"Found {len(outliers)} out-of-control points in training data")
                            st.dataframe(outliers, use_container_width=True)
                        else:
                            st.success("No out-of-control points detected in training data")
                    except Exception as e:
                        st.error(f"Error detecting outliers: {str(e)}")
            
            with col2:
                if mpls.X_test is not None and st.button("Detect Test Outliers"):
                    try:
                        outliers = mpls.detect_outliers(use_train=False)
                        if outliers is not None and len(outliers) > 0:
                            st.warning(f"Found {len(outliers)} out-of-control points in test data")
                            st.dataframe(outliers, use_container_width=True)
                        else:
                            st.success("No out-of-control points detected in test data")
                    except Exception as e:
                        st.error(f"Error detecting outliers: {str(e)}")
            
            with st.expander("View Monitoring Statistics"):
                tab1, tab2 = st.tabs(["Training Stats", "Test Stats"])
                
                with tab1:
                    stats_train = mpls.get_monitoring_stats(use_train=True)
                    if stats_train is not None:
                        st.dataframe(stats_train, use_container_width=True)
                        
                        csv_train = stats_train.to_csv(index=False)
                        st.download_button(
                            "Download Training Stats CSV",
                            csv_train,
                            "mpls_training_stats.csv",
                            "text/csv"
                        )
                
                with tab2:
                    if mpls.X_test is not None:
                        stats_test = mpls.get_monitoring_stats(use_train=False)
                        if stats_test is not None:
                            st.dataframe(stats_test, use_container_width=True)
                            
                            csv_test = stats_test.to_csv(index=False)
                            st.download_button(
                                "Download Test Stats CSV",
                                csv_test,
                                "mpls_test_stats.csv",
                                "text/csv"
                            )
                    else:
                        st.info("No test data available")
            
            with st.expander("View PLS Scores"):
                tab1, tab2 = st.tabs(["Training Scores", "Test Scores"])
                
                with tab1:
                    scores_train = mpls.get_scores(use_train=True)
                    if scores_train is not None:
                        st.dataframe(scores_train, use_container_width=True)
                
                with tab2:
                    if mpls.X_test is not None:
                        scores_test = mpls.get_scores(use_train=False)
                        if scores_test is not None:
                            st.dataframe(scores_test, use_container_width=True)
                    else:
                        st.info("No test data available")
            
            with st.expander("View PLS Loadings"):
                loadings = mpls.get_loadings()
                if loadings is not None:
                    st.dataframe(loadings, use_container_width=True)
                    
                    csv_loadings = loadings.to_csv()
                    st.download_button(
                        "Download Loadings CSV",
                        csv_loadings,
                        "mpls_loadings.csv",
                        "text/csv"
                    )
            
            st.session_state.mpls_done = True
        
        st.divider()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button(" Back to Augmentation"):
                st.session_state.step = 8
                st.rerun()
        
        with col2:
            if st.session_state.mpls_model is not None:
                if st.button("Proceed to Model Training ", type="primary"):
                    st.session_state.step = 10
                    st.rerun()
        
        with col3:
            if st.button("Skip MPLS"):
                st.session_state.step = 9
                st.info("Skipped MPLS analysis")
                st.rerun()
    
#################################################################################################################################################


    elif st.session_state.step == 10:
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            mean_absolute_error,
            mean_squared_error,
            r2_score,
        )

        st.header("Step 10: Model Training")

        if not getattr(st.session_state, "data_split", False):
            st.error("Please complete train-test split first")
            return

        def flatten_y(y):
            if y is None:
                return None
            if hasattr(y, "values"):
                yv = y.values
            else:
                yv = np.asarray(y)
            return yv.ravel() if yv.ndim > 1 else yv

        X_train = X_test = y_train = y_test = None
        if (
            "augmented_X_train" in st.session_state
            and "augmented_y_train" in st.session_state
            and st.session_state.augmented_X_train is not None
            and st.session_state.augmented_y_train is not None
            and getattr(st.session_state.augmented_X_train, "size", 0) > 0
            and getattr(st.session_state.augmented_y_train, "size", 0) > 0
        ):
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
            except Exception:
                X_train = None

        if X_train is None:
            if all(
                hasattr(st.session_state, k)
                for k in ("X_train", "X_test", "y_train", "y_test")
            ):
                X_train = st.session_state.X_train
                X_test = st.session_state.X_test
                y_train = st.session_state.y_train
                y_test = st.session_state.y_test
                st.warning("Using original (non-augmented) data as fallback")
            else:
                st.error("No training data available. Please complete previous steps.")
                return

        y_train = flatten_y(y_train)
        y_test = flatten_y(y_test)

        st.write(f"Training data ready: Train={X_train.shape}, Test={X_test.shape}")
        st.write(f"Target shapes: y_train={y_train.shape}, y_test={y_test.shape}")

        if (
            "augmented_X_train" in st.session_state
            and st.session_state.augmented_X_train is not None
            and getattr(st.session_state.augmented_X_train, "size", 0) > 0
            and X_train.shape[0] == st.session_state.augmented_X_train.shape[0]
        ):
            original_size = (
                st.session_state.X_train.shape[0]
                if hasattr(st.session_state, "X_train")
                and hasattr(st.session_state.X_train, "shape")
                else "Unknown"
            )
            augmented_size = X_train.shape[0]
            st.success(
                f"Augmented Training Data Active: {original_size} → {augmented_size} samples"
            )
            if getattr(st.session_state, "augmentation_history", None):
                with st.expander("View Augmentation History"):
                    for i, aug in enumerate(st.session_state.augmentation_history):
                        st.write(f"{i+1}. {aug}")

        # Shape consistency
        if X_train.shape[0] != y_train.shape[0] or X_test.shape[0] != y_test.shape[0]:
            st.error("Data inconsistency detected!")
            st.error(
                f"Training: X={X_train.shape[0]} samples, y={y_train.shape[0]} targets"
            )
            st.error(f"Testing: X={X_test.shape[0]} samples, y={y_test.shape[0]} targets")
            return

        model_type = st.radio(
            "Select model type:", ("Normal Modeling", "Zero-Inflated Modeling")
        )

        
        if model_type == "Normal Modeling":
            model_flow = st.radio("Select model run type", ("Manual", "Optuna Tuning", "Automated"))
            task_type = st.radio("Select task type:", ("Regression", "Classification"))

            if model_flow == "Manual":
                if task_type == "Regression":
                    model_choice = st.selectbox(
                        "Select regression model:",
                        [
                            "Linear Regression",
                            "Lasso Regression",
                            "Ridge Regression",
                            "ElasticNet Regression",
                            "Decision Tree",
                            "Random Forest",
                            "Gradient Boosting",
                            "AdaBoost",
                            "SVR",
                            "XGBoost Regressor",
                        ],
                    )
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

                            model, predictions, r2_val, mae, mse, rmse = results

                            st.session_state.trained_model = model
                            st.session_state.model_parameters = {
                                "model_name": model_choice,
                                "model_type": "regression",
                                "training_method": "manual",
                                "random_state": 42,
                                "performance_metrics": {
                                    "r2_score": r2_val,
                                    "mae": mae,
                                    "mse": mse,
                                    "rmse": rmse,
                                },
                            }
                            st.session_state.predictions = predictions
                            st.session_state.model_trained = True

                            st.success("Model trained successfully!")
                            st.write(f"R² Score: {r2_val:.4f}")
                            st.write(f"MAE: {mae:.4f}")
                            st.write(f"MSE: {mse:.4f}")
                            st.write(f"RMSE: {rmse:.4f}")

                        except Exception as e:
                            st.error(f"Error in model training: {str(e)}")
                else:
                    model_choice = st.selectbox(
                        "Select classification model:",
                        [
                            "Logistic Regression",
                            "Decision Tree",
                            "Random Forest",
                            "Gradient Boosting",
                            "AdaBoost",
                            "SVM Classifier",
                            "KNN Classifier",
                            "XGBoost Classifier",
                        ],
                    )
                    if st.button("Run Model"):
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
                            elif model_choice == "SVM Classifier":
                                results = manual.SVR_classifier(X_train, X_test, y_train, y_test)
                            elif model_choice == "KNN Classifier":
                                results = manual.KNN_classifier(X_train, X_test, y_train, y_test)
                            elif model_choice == "XGBoost Classifier":
                                results = manual.XGBoost_classifier(X_train, X_test, y_train, y_test)

                            model, predictions, accuracy, f1_val = results

                            st.session_state.trained_model = model
                            st.session_state.model_parameters = {
                                "model_name": model_choice,
                                "model_type": "classification",
                                "training_method": "manual",
                                "performance_metrics": {
                                    "accuracy": accuracy,
                                    "f1_score": f1_val,
                                },
                            }
                            st.session_state.predictions = predictions
                            st.session_state.model_trained = True

                            st.success("Model trained successfully!")
                            st.write(f"Accuracy: {accuracy:.4f}")
                            st.write(f"F1 Score: {f1_val:.4f}")

                        except Exception as e:
                            st.error(f"Error in model training: {str(e)}")

            elif model_flow == "Optuna Tuning":
                if task_type == "Regression":
                    model_choice = st.selectbox(
                        "Select regression model:",
                        [
                            "Ridge Regression",
                            "ElasticNet Regression",
                            "Decision Tree",
                            "Random Forest",
                            "Gradient Boosting",
                            "AdaBoost",
                            "SVR",
                            "XGBoost Regressor",
                            "KNN Regressor",
                            "GaussianProcessRegressor",
                            "Ensemble Model",
                            "PLS Regression",
                            "ANN_regression",
                        ],
                    )
                    if st.button("Run Model"):
                        try:
                            if model_choice == "Ridge Regression":
                                results = optuna.Ridge_regression(X_train, X_test, y_train, y_test)
                            elif model_choice == "ElasticNet Regression":
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

                            model, predictions, r2_val, mae, mse, rmse, study = results

                            st.session_state.trained_model = model
                            st.session_state.model_parameters = {
                                "model_name": model_choice,
                                "model_type": "regression",
                                "training_method": "optuna",
                                "random_state": 42,
                                "performance_metrics": {
                                    "r2_score": r2_val,
                                    "mae": mae,
                                    "mse": mse,
                                    "rmse": rmse,
                                },
                            }
                            st.session_state.predictions = predictions
                            st.session_state.model_trained = True

                            st.success("Model trained successfully!")
                            st.write(f"R² Score: {r2_val:.4f}")
                            st.write(f"MAE: {mae:.4f}")
                            st.write(f"MSE: {mse:.4f}")
                            st.write(f"RMSE: {rmse:.4f}")

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
                    model_choice = st.selectbox(
                        "Select classification model:",
                        ["Logistic Regression", "Random Forest", "XGBoost Classifier"],
                    )
                    if st.button("Run Model"):
                        try:
                            if model_choice == "Logistic Regression":
                                results = optuna.Logistic_regression(X_train, X_test, y_train, y_test)
                            elif model_choice == "Random Forest":
                                results = optuna.Random_forest_classifier(X_train, X_test, y_train, y_test)
                            elif model_choice == "XGBoost Classifier":
                                results = optuna.XGBoost_classifier(X_train, X_test, y_train, y_test)

                            model, predictions, accuracy, f1_val = results

                            st.session_state.trained_model = model
                            st.session_state.model_parameters = {
                                "model_name": model_choice,
                                "model_type": "classification",
                                "training_method": "optuna",
                                "performance_metrics": {
                                    "accuracy": accuracy,
                                    "f1_score": f1_val,
                                },
                            }
                            st.session_state.predictions = predictions
                            st.session_state.model_trained = True

                            st.success("Model trained successfully!")
                            st.write(f"Accuracy: {accuracy:.4f}")
                            st.write(f"F1 Score: {f1_val:.4f}")

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
                            "model_name": "Automated Selection",
                            "model_type": task_type.lower(),
                            "training_method": "automated",
                        }
                        st.session_state.model_trained = True
                        st.success("Automated model selection completed!")
                    except Exception as e:
                        st.error(f"Error in automated model selection: {str(e)}")

       
        else:
            y_train_flat = flatten_y(y_train)
            y_test_flat = flatten_y(y_test)

            st.write(f"Training data ready: Train={X_train.shape}, Test={X_test.shape}")
            st.write(f"Target shapes: y_train={y_train_flat.shape}, y_test={y_test_flat.shape}")

            st.subheader("Zero-Inflation Analysis")
            zero_count_train = int(np.sum(y_train_flat == 0))
            nonzero_count_train = int(np.sum(y_train_flat != 0))
            zero_percentage = (zero_count_train / max(1, len(y_train_flat))) * 100.0

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
                st.success("Good candidate for zero-inflated modeling!")

            if (
                "augmented_X_train" in st.session_state
                and st.session_state.augmented_X_train is not None
                and getattr(st.session_state.augmented_X_train, "size", 0) > 0
                and X_train.shape[0] == st.session_state.augmented_X_train.shape[0]
            ):
                original_size = (
                    st.session_state.X_train.shape[0]
                    if hasattr(st.session_state, "X_train")
                    and hasattr(st.session_state.X_train, "shape")
                    else "Unknown"
                )
                augmented_size = X_train.shape[0]
                st.success(f"Augmented Training Data Active: {original_size} → {augmented_size} samples")
                if getattr(st.session_state, "augmentation_history", None):
                    with st.expander("View Augmentation History"):
                        for i, aug in enumerate(st.session_state.augmentation_history):
                            st.write(f"{i+1}. {aug}")

            if X_train.shape[0] != y_train_flat.shape[0] or X_test.shape[0] != y_test_flat.shape[0]:
                st.error("Data inconsistency detected!")
                st.error(f"Training: X={X_train.shape[0]} samples, y={y_train_flat.shape[0]} targets")
                st.error(f"Testing: X={X_test.shape[0]} samples, y={y_test_flat.shape[0]} targets")
                return

            y_binary_train = (y_train_flat != 0).astype(int)
            y_binary_test = (y_test_flat != 0).astype(int)

            nonzero_mask_train = y_train_flat != 0
            nonzero_mask_test = y_test_flat != 0

            if not np.any(nonzero_mask_train):
                st.error("No non-zero values in training target. Zero-inflated regression cannot proceed.")
                return

            X_reg_train = X_train[nonzero_mask_train]
            y_reg_train = y_train_flat[nonzero_mask_train]
            X_reg_test = X_test[nonzero_mask_test]
            y_reg_test = y_test_flat[nonzero_mask_test]

            st.write(f"Binary classification data: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
            st.write(
                f"Regression data (non-zero only): Train={X_reg_train.shape[0]}, Test={X_reg_test.shape[0]}"
            )

            model_flow = st.radio("Select model run type", ("Manual", "Optuna Tuning", "Automated"))

            if "two_part_models" not in st.session_state:
                st.session_state.two_part_models = {}

            def create_combined_predictions_and_metrics(
                clf_model,
                reg_model,
                X_train,
                X_test,
                y_train_flat,
                y_test_flat,
                y_binary_train,
                y_binary_test,
                X_reg_train,
                y_reg_train,
                X_reg_test,
                y_reg_test,
            ):
                clf_train_pred = clf_model.predict(X_train)
                clf_test_pred = clf_model.predict(X_test)

                reg_train_pred = reg_model.predict(X_reg_train) if len(X_reg_train) > 0 else np.array([])
                reg_test_pred = reg_model.predict(X_reg_test) if len(X_reg_test) > 0 else np.array([])

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

                if len(y_reg_train) > 0 and len(reg_train_pred) > 0:
                    reg_train_r2 = r2_score(y_reg_train, reg_train_pred)
                    reg_train_mae = mean_absolute_error(y_reg_train, reg_train_pred)
                    reg_train_rmse = np.sqrt(mean_squared_error(y_reg_train, reg_train_pred))
                else:
                    reg_train_r2 = np.nan
                    reg_train_mae = np.nan
                    reg_train_rmse = np.nan

                if len(y_reg_test) > 0 and len(reg_test_pred) > 0:
                    reg_test_r2 = r2_score(y_reg_test, reg_test_pred)
                    reg_test_mae = mean_absolute_error(y_reg_test, reg_test_pred)
                    reg_test_rmse = np.sqrt(mean_squared_error(y_reg_test, reg_test_pred))
                else:
                    reg_test_r2 = np.nan
                    reg_test_mae = np.nan
                    reg_test_rmse = np.nan

                def safe_r2(y_true, y_pred):
                    try:
                        return r2_score(y_true, y_pred)
                    except Exception:
                        return np.nan

                overall_train_r2 = safe_r2(y_train_flat, combined_train_pred)
                overall_test_r2 = safe_r2(y_test_flat, combined_test_pred)
                overall_train_mae = mean_absolute_error(y_train_flat, combined_train_pred)
                overall_test_mae = mean_absolute_error(y_test_flat, combined_test_pred)
                overall_train_rmse = np.sqrt(mean_squared_error(y_train_flat, combined_train_pred))
                overall_test_rmse = np.sqrt(mean_squared_error(y_test_flat, combined_test_pred))

                return {
                    "train_metrics": {
                        "classification": {"accuracy": clf_train_acc, "f1_score": clf_train_f1},
                        "regression": {
                            "r2_score": reg_train_r2,
                            "mae": reg_train_mae,
                            "rmse": reg_train_rmse,
                        },
                        "overall": {
                            "r2_score": overall_train_r2,
                            "mae": overall_train_mae,
                            "rmse": overall_train_rmse,
                        },
                    },
                    "test_metrics": {
                        "classification": {"accuracy": clf_test_acc, "f1_score": clf_test_f1},
                        "regression": {
                            "r2_score": reg_test_r2,
                            "mae": reg_test_mae,
                            "rmse": reg_test_rmse,
                        },
                        "overall": {
                            "r2_score": overall_test_r2,
                            "mae": overall_test_mae,
                            "rmse": overall_test_rmse,
                        },
                    },
                    "predictions": {
                        "binary_test": clf_test_pred,
                        "regression_test": reg_test_pred,
                        "combined_test": combined_test_pred,
                        "combined_train": combined_train_pred,
                    },
                }

            def plot_train_test_comparison(train_metrics, test_metrics):
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

                categories = ["Classification\nAccuracy", "Classification\nF1 Score"]
                train_clf = [
                    train_metrics["classification"]["accuracy"],
                    train_metrics["classification"]["f1_score"],
                ]
                test_clf = [
                    test_metrics["classification"]["accuracy"],
                    test_metrics["classification"]["f1_score"],
                ]

                x = np.arange(len(categories))
                width = 0.35

                ax1.bar(x - width / 2, train_clf, width, label="Train", color="skyblue")
                ax1.bar(x + width / 2, test_clf, width, label="Test", color="lightcoral")
                ax1.set_ylabel("Score")
                ax1.set_title("Classification Performance")
                ax1.set_xticks(x)
                ax1.set_xticklabels(categories)
                ax1.legend()
                ax1.set_ylim(0, 1)

                reg_categories = ["R² Score", "MAE", "RMSE"]
                train_reg = [
                    train_metrics["regression"]["r2_score"],
                    train_metrics["regression"]["mae"],
                    train_metrics["regression"]["rmse"],
                ]
                test_reg = [
                    test_metrics["regression"]["r2_score"],
                    test_metrics["regression"]["mae"],
                    test_metrics["regression"]["rmse"],
                ]

                x2 = np.arange(len(reg_categories))
                ax2.bar(x2 - width / 2, train_reg, width, label="Train", color="skyblue")
                ax2.bar(x2 + width / 2, test_reg, width, label="Test", color="lightcoral")
                ax2.set_ylabel("Score")
                ax2.set_title("Regression Performance (Non-Zero Values)")
                ax2.set_xticks(x2)
                ax2.set_xticklabels(reg_categories)
                ax2.legend()

                overall_categories = ["R² Score", "MAE", "RMSE"]
                train_overall = [
                    train_metrics["overall"]["r2_score"],
                    train_metrics["overall"]["mae"],
                    train_metrics["overall"]["rmse"],
                ]
                test_overall = [
                    test_metrics["overall"]["r2_score"],
                    test_metrics["overall"]["mae"],
                    test_metrics["overall"]["rmse"],
                ]

                x3 = np.arange(len(overall_categories))
                ax3.bar(x3 - width / 2, train_overall, width, label="Train", color="skyblue")
                ax3.bar(x3 + width / 2, test_overall, width, label="Test", color="lightcoral")
                ax3.set_ylabel("Score")
                ax3.set_title("Overall Combined Performance")
                ax3.set_xticks(x3)
                ax3.set_xticklabels(overall_categories)
                ax3.legend()

                r2_comparison = [train_metrics["overall"]["r2_score"], test_metrics["overall"]["r2_score"]]
                ax4.bar(["Train R²", "Test R²"], r2_comparison, color=["skyblue", "lightcoral"])
                ax4.set_ylabel("R² Score")
                ax4.set_title("Train vs Test R² Comparison")
                ymax = np.nanmax([1, *[v for v in r2_comparison if v == v]])  
                ax4.set_ylim(0, max(1, ymax * 1.1))

                for i, v in enumerate(r2_comparison):
                    if v == v:  # not NaN
                        ax4.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontweight="bold")

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            # Two-part flows
            if model_flow == "Manual":
                st.subheader("Model Selection")

                col1, col2 = st.columns(2)
                with col1:
                    st.write("Step 1: Classification Model (Zero vs Non-Zero)")
                    classification_model = st.selectbox(
                        "Select classification model:",
                        [
                            "Logistic Regression",
                            "Decision Tree",
                            "Random Forest",
                            "Gradient Boosting",
                            "AdaBoost",
                            "SVM Classifier",
                            "XGBoost Classifier",
                        ],
                        key="clf_model",
                    )

                with col2:
                    st.write("Step 2: Regression Model (Non-Zero Values)")
                    regression_model = st.selectbox(
                        "Select regression model:",
                        [
                            "Linear Regression",
                            "Lasso Regression",
                            "Ridge Regression",
                            "ElasticNet Regression",
                            "Decision Tree",
                            "Random Forest",
                            "Gradient Boosting",
                            "AdaBoost",
                            "SVR",
                            "XGBoost Regressor",
                        ],
                        key="reg_model",
                    )

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
                        st.success(f"Classification Model trained! Accuracy: {clf_accuracy:.4f}, F1: {clf_f1:.4f}")

                        st.write("Training Regression Model...")
                        if len(X_reg_train) == 0:
                            st.error("No non-zero training data available for regression!")
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
                        st.success(f"Regression Model trained! R²: {reg_r2:.4f}, RMSE: {reg_rmse:.4f}")

                        st.write("Combining predictions...")
                        metrics_data = create_combined_predictions_and_metrics(
                            clf_model,
                            reg_model,
                            X_train,
                            X_test,
                            y_train_flat,
                            y_test_flat,
                            y_binary_train,
                            y_binary_test,
                            X_reg_train,
                            y_reg_train,
                            X_reg_test,
                            y_reg_test,
                        )

                        st.session_state.two_part_models = {
                            "classification_model": clf_model,
                            "regression_model": reg_model,
                            "classification_name": classification_model,
                            "regression_name": regression_model,
                            "training_method": "manual",
                            "performance_metrics": metrics_data["test_metrics"],
                            "train_metrics": metrics_data["train_metrics"],
                            "predictions": metrics_data["predictions"],
                        }

                        st.session_state.trained_model = st.session_state.two_part_models
                        st.session_state.model_parameters = {
                            "model_name": "Zero-Inflated Model",
                            "model_type": "zero_inflated",
                            "training_method": "manual",
                            "classification_model": classification_model,
                            "regression_model": regression_model,
                            "performance_metrics": metrics_data["test_metrics"],
                        }
                        st.session_state.predictions = metrics_data["predictions"]["combined_test"]
                        st.session_state.model_trained = True

                        st.subheader("Two-Part Model Results")
                        plot_train_test_comparison(
                            metrics_data["train_metrics"], metrics_data["test_metrics"]
                        )

                    except Exception as e:
                        st.error(f"Error in two-part model training: {str(e)}")
                        import traceback

                        st.error(f"Traceback: {traceback.format_exc()}")

            elif model_flow == "Optuna Tuning":
                st.subheader("Optuna-Optimized Two-Part Model")

                col1, col2 = st.columns(2)
                with col1:
                    st.write("Step 1: Classification Model (Zero vs Non-Zero)")
                    classification_model = st.selectbox(
                        "Select classification model:",
                        ["Logistic Regression", "Random Forest", "XGBoost Classifier"],
                        key="optuna_clf_model",
                    )

                with col2:
                    st.write("Step 2: Regression Model (Non-Zero Values)")
                    regression_model = st.selectbox(
                        "Select regression model:",
                        [
                            "Ridge Regression",
                            "ElasticNet Regression",
                            "Decision Tree",
                            "Random Forest",
                            "Gradient Boosting",
                            "AdaBoost",
                            "SVR",
                            "XGBoost Regressor",
                        ],
                        key="optuna_reg_model",
                    )

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
                        st.success(f"Classification Model trained! Accuracy: {clf_accuracy:.4f}, F1: {clf_f1:.4f}")

                        st.write("Training Regression Model with Optuna...")
                        if len(X_reg_train) == 0:
                            st.error("No non-zero training data available for regression!")
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

                        st.success(f"Regression Model trained! R²: {reg_r2:.4f}, RMSE: {reg_rmse:.4f}")

                        st.write("Combining predictions...")
                        metrics_data = create_combined_predictions_and_metrics(
                            clf_model,
                            reg_model,
                            X_train,
                            X_test,
                            y_train_flat,
                            y_test_flat,
                            y_binary_train,
                            y_binary_test,
                            X_reg_train,
                            y_reg_train,
                            X_reg_test,
                            y_reg_test,
                        )

                        st.session_state.two_part_models = {
                            "classification_model": clf_model,
                            "regression_model": reg_model,
                            "classification_name": classification_model,
                            "regression_name": regression_model,
                            "training_method": "optuna",
                            "performance_metrics": metrics_data["test_metrics"],
                            "train_metrics": metrics_data["train_metrics"],
                            "predictions": metrics_data["predictions"],
                        }

                        st.session_state.trained_model = st.session_state.two_part_models
                        st.session_state.model_parameters = {
                            "model_name": "Zero-Inflated Model (Optuna)",
                            "model_type": "zero_inflated",
                            "training_method": "optuna",
                            "classification_model": classification_model,
                            "regression_model": regression_model,
                            "performance_metrics": metrics_data["test_metrics"],
                        }
                        st.session_state.predictions = metrics_data["predictions"]["combined_test"]
                        st.session_state.model_trained = True

                        st.subheader("Optuna Two-Part Model Results")
                        plot_train_test_comparison(
                            metrics_data["train_metrics"], metrics_data["test_metrics"]
                        )

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
                            st.error("No non-zero training data available for regression!")
                            return

                        best_reg_model = auto.run_regression(
                            X_reg_train, X_reg_test, y_reg_train, y_reg_test
                        )

                        st.write("Combining automated models...")
                        metrics_data = create_combined_predictions_and_metrics(
                            best_clf_model,
                            best_reg_model,
                            X_train,
                            X_test,
                            y_train_flat,
                            y_test_flat,
                            y_binary_train,
                            y_binary_test,
                            X_reg_train,
                            y_reg_train,
                            X_reg_test,
                            y_reg_test,
                        )

                        st.session_state.two_part_models = {
                            "classification_model": best_clf_model,
                            "regression_model": best_reg_model,
                            "classification_name": "Automated Selection",
                            "regression_name": "Automated Selection",
                            "training_method": "automated",
                            "performance_metrics": metrics_data["test_metrics"],
                            "train_metrics": metrics_data["train_metrics"],
                            "predictions": metrics_data["predictions"],
                        }

                        st.session_state.trained_model = st.session_state.two_part_models
                        st.session_state.model_parameters = {
                            "model_name": "Zero-Inflated Model (Automated)",
                            "model_type": "zero_inflated",
                            "training_method": "automated",
                            "classification_model": "Automated Selection",
                            "regression_model": "Automated Selection",
                            "performance_metrics": metrics_data["test_metrics"],
                        }
                        st.session_state.predictions = metrics_data["predictions"]["combined_test"]
                        st.session_state.model_trained = True

                        st.success("Automated two-part model selection completed!")

                        st.subheader("Automated Two-Part Model Results")
                        plot_train_test_comparison(
                            metrics_data["train_metrics"], metrics_data["test_metrics"]
                        )

                    except Exception as e:
                        st.error(f"Error in automated two-part model selection: {str(e)}")
                        import traceback

                        st.error(f"Traceback: {traceback.format_exc()}")

        if getattr(st.session_state, "model_trained", False):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Proceed to Evaluation"):
                    st.session_state.step = 11
                    st.rerun()
            with col2:
                if st.button("Skip to Model Saving"):
                    st.session_state.skipped_steps.add(10)
                    st.session_state.step = 12
                    st.rerun()

#####################################################################################################################################################################

    elif st.session_state.step == 11:
        st.header("Step 11: Model Evaluation")
        
        if not st.session_state.model_trained:
            st.error("Please train a model first")
            if st.button("Go back to Model Training"):
                st.session_state.step = 10
                st.rerun()
            return
        
        model = st.session_state.trained_model
        model_params = st.session_state.model_parameters
        
        st.subheader("Model Performance Summary")
        
        if 'performance_metrics' in model_params:
            metrics = model_params['performance_metrics']
            
            if model_params.get('model_type') == 'regression':
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("R² Score", f"{metrics.get('r2_score', 0):.4f}")
                with col2:
                    st.metric("MAE", f"{metrics.get('mae', 0):.4f}")
                with col3:
                    st.metric("MSE", f"{metrics.get('mse', 0):.4f}")
                with col4:
                    st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
                with col2:
                    st.metric("F1 Score", f"{metrics.get('f1_score', 0):.4f}")
        
        if 'predictions' in st.session_state:
            st.subheader("Prediction vs Actual Plot")
            try:
                predictions = st.session_state.predictions
                y_test = st.session_state.y_test
                
                if hasattr(y_test, 'values'):
                    y_test_vals = y_test.values.ravel() if len(y_test.shape) > 1 else y_test.values
                else:
                    y_test_vals = y_test.ravel() if len(y_test.shape) > 1 else y_test
                
                pred_vals = predictions.ravel() if len(predictions.shape) > 1 else predictions
                import matplotlib.pyplot as plt 
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(y_test_vals, pred_vals, alpha=0.6)
                ax.plot([y_test_vals.min(), y_test_vals.max()], [y_test_vals.min(), y_test_vals.max()], 'r--', lw=2)
                ax.set_xlabel('Actual Values')
                ax.set_ylabel('Predicted Values')
                ax.set_title('Prediction vs Actual Values')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)
                
            except Exception as plot_error:
                st.warning(f"Could not create prediction plot: {plot_error}")
        
        st.session_state.model_evaluated = True
        
        if st.button("Proceed to Model Saving"):
            st.session_state.step = 12
            st.rerun()

#####################################################################################################################################################################

    # Step 11: Model Saving
    elif st.session_state.step == 12:
        st.header("Step 12: Model Saving")
        import numpy as np
        if not st.session_state.model_trained:
            st.error("Please train a model first")
            if st.button("Go back to Model Training"):
                st.session_state.step = 9
                st.rerun()
            return
        
        if st.session_state.get('user_name'):
            st.subheader("User Information")
            st.success(f"Model will be saved for: {st.session_state.user_name}")
        else:
            st.warning("No user name provided.")
            if st.button("Go back to Step 1"):
                st.session_state.step = 1
                st.rerun()
            return
        
        st.subheader("Model Information")
        if st.session_state.get('model_parameters'):
            st.write(f"Model Name: {st.session_state.model_parameters.get('model_name', 'Unknown')}")
            st.write(f"Model Type: {st.session_state.model_parameters.get('model_type', 'Unknown')}")
            st.write(f"Training Method: {st.session_state.model_parameters.get('training_method', 'Unknown')}")
            
            if 'performance_metrics' in st.session_state.model_parameters:
                st.write("Performance Metrics:")
                for metric, value in st.session_state.model_parameters['performance_metrics'].items():
                    if isinstance(value, (int, float)):
                        st.write(f"  - {metric}: {value:.4f}")
                    else:
                        st.write(f"  - {metric}: {value}")
        
        st.subheader("Save Model and Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            save_model_name = st.text_input("Custom Model Name (optional)", 
                                        value=st.session_state.get('model_parameters', {}).get('model_name', 'model'))
        with col2:
            include_preprocessing = st.checkbox("Include preprocessing parameters", value=True)
        
        if st.button("Save Model and Parameters"):
            try:
                model_params = st.session_state.get('model_parameters', {}).copy()
                
                if 'X_train' in st.session_state and hasattr(st.session_state.X_train, 'shape') and not hasattr(st.session_state.X_train, 'columns'):
                    feature_names = [f'feature_{i}' for i in range(st.session_state.X_train.shape[1])]
                    st.session_state.X_train = pd.DataFrame(st.session_state.X_train, columns=feature_names)
                
                preprocessing_params = {}
                if include_preprocessing:
                    preprocessing_params = st.session_state.get('preprocessing_parameters', {}).copy()
                
                model_filename, json_filename = save_model_and_parameters(
                    st.session_state.get('trained_model'),
                    save_model_name,
                    model_params,
                    preprocessing_params
                )
                
                st.session_state.model_saved = True
                st.session_state.saved_model_filename = model_filename
                st.session_state.saved_json_filename = json_filename
                
                # Check if fitted objects file was created
                fitted_objects_filename = None
                if os.path.exists(json_filename):
                    with open(json_filename, 'r') as f:
                        saved_params = json.load(f)
                    fitted_objects_filename = saved_params.get('model_info', {}).get('fitted_objects_filename')
                st.session_state.saved_fitted_objects_filename = fitted_objects_filename
                
                st.success("Model and parameters saved successfully!")
                st.write(f"User: {st.session_state.user_name}")
                st.write(f"Model file: {model_filename}")
                st.write(f"Parameters file: {json_filename}")
                if fitted_objects_filename:
                    st.write(f"Fitted objects file: {fitted_objects_filename}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if os.path.exists(model_filename):
                        with open(model_filename, 'rb') as f:
                            st.download_button(
                                "Download Model File",
                                data=f.read(),
                                file_name=os.path.basename(model_filename),
                                mime="application/octet-stream"
                            )
                
                with col2:
                    if os.path.exists(json_filename):
                        with open(json_filename, 'r') as f:
                            st.download_button(
                                "Download Parameters JSON",
                                data=f.read(),
                                file_name=os.path.basename(json_filename),
                                mime="application/json"
                            )
                
                with col3:
                    if fitted_objects_filename and os.path.exists(fitted_objects_filename):
                        with open(fitted_objects_filename, 'rb') as f:
                            st.download_button(
                                "Download Fitted Objects",
                                data=f.read(),
                                file_name=os.path.basename(fitted_objects_filename),
                                mime="application/octet-stream"
                            )
                
            except Exception as e:
                st.error(f"Error saving model: {str(e)}")
        
        if st.session_state.get('model_saved'):
            st.subheader("Parameter Summary")
            
            summary = {
                "User Information": {
                    "user_name": st.session_state.get('user_name', 'Unknown'),
                    "creation_date": datetime.now().isoformat()
                },
                "Model Information": st.session_state.get('model_parameters', {}),
                "Preprocessing Parameters": st.session_state.get('preprocessing_parameters', {}) if include_preprocessing else "Not included"
            }
            
            st.json(summary)
        
        if st.session_state.get('model_saved'):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Proceed to New Data Preprocessing"):
                    st.session_state.step = 13
                    st.rerun()
            with col2:
                if st.button("Go Back to Evaluation"):
                    st.session_state.step = 11
                    st.rerun()

#####################################################################################################################################################################

    elif st.session_state.step == 13:
        st.header("Step 13: Preprocessing on New Data")

        if st.session_state.direct_prediction_mode:
            st.info("Direct Prediction Mode - Load your saved model and preprocess new data")
            
            st.subheader("Load Saved Model and Parameters")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                uploaded_model = st.file_uploader("Upload saved model file (.pkl)", type=["pkl"], key="direct_model")
            with col2:
                uploaded_params = st.file_uploader("Upload parameters JSON file", type=["json"], key="direct_params")
            with col3:
                uploaded_fitted = st.file_uploader("Upload fitted objects (.pkl, optional)", type=["pkl"], key="direct_fitted")
            
            if uploaded_model and uploaded_params:
                try:
                    st.session_state.loaded_model = pickle.load(uploaded_model)
                    
                    params_data = json.load(uploaded_params)
                    st.session_state.loaded_parameters = params_data
                    
                    # Load fitted objects if provided
                    if uploaded_fitted:
                        st.session_state.loaded_fitted_objects = pickle.load(uploaded_fitted)
                        st.success("Fitted preprocessing objects loaded!")
                    elif 'loaded_fitted_objects' not in st.session_state:
                        # Try loading from same directory based on JSON reference
                        fitted_filename = params_data.get('model_info', {}).get('fitted_objects_filename')
                        if fitted_filename and os.path.exists(fitted_filename):
                            with open(fitted_filename, 'rb') as f:
                                st.session_state.loaded_fitted_objects = pickle.load(f)
                            st.success("Fitted preprocessing objects auto-loaded!")
                    
                    saved_user = params_data.get('user_info', {}).get('user_name', 'Unknown')
                    if saved_user != 'Unknown':
                        if saved_user == st.session_state.user_name:
                            st.success(f"Model loaded successfully! Created by: {saved_user}")
                        else:
                            st.warning(f"Model was created by: {saved_user} (You are: {st.session_state.user_name})")
                    
                    model_info = params_data.get('model_info', {})
                    st.write("Loaded Model Information:")
                    st.write(f"- Model Name: {model_info.get('model_name', 'Unknown')}")
                    st.write(f"- Created: {params_data.get('user_info', {}).get('creation_date', 'Unknown')}")
                    st.write(f"- File: {model_info.get('model_filename', 'Unknown')}")
                    
                    model_params = params_data.get('model_parameters', {})
                    if 'performance_metrics' in model_params:
                        st.write("Model Performance:")
                        for metric, value in model_params['performance_metrics'].items():
                            if isinstance(value, (int, float)):
                                st.write(f"  - {metric}: {value:.4f}")
                    
                except Exception as e:
                    st.error(f"Error loading model/parameters: {str(e)}")
        
        model_available = (not st.session_state.direct_prediction_mode and st.session_state.get("model_saved", False)) or \
                        (st.session_state.direct_prediction_mode and 'loaded_model' in st.session_state)
        
        if not model_available:
            if st.session_state.direct_prediction_mode:
                st.error("Please load your saved model and parameters first")
                return
            else:
                st.error("Please save your model first")
                if st.button("Go back to Model Saving"):
                    st.session_state.step = 12
                    st.rerun()
                return

        st.subheader("Load and Preprocess New Data")

        st.markdown("### Upload New Data for Prediction")
        new_data_file = st.file_uploader("Upload new data for prediction", type=["csv", "xlsx", "txt"])

        if new_data_file:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{new_data_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(new_data_file.getbuffer())
                    file_path = tmp_file.name

                new_data = rd.read_data(file_path)
                st.session_state.new_data = new_data.copy()

                st.success("New data loaded successfully!")
                st.write(f"Data shape: {new_data.shape}")
                display_new_data = new_data.copy()
                for col in display_new_data.columns:
                    if display_new_data[col].dtype == 'int64':
                        display_new_data[col] = display_new_data[col].astype('int32')
                    elif display_new_data[col].dtype == 'float64':
                        display_new_data[col] = display_new_data[col].astype('float32')
                st.dataframe(display_new_data.head())

                os.unlink(file_path)

            except Exception as e:
                st.error(f"Error loading new data: {str(e)}")

        if 'processed_new_data' in st.session_state:
            st.success("New data has been preprocessed and is ready for prediction!")
            st.write(f"Preprocessed data shape: {st.session_state.processed_new_data.shape}")
        else:
            st.info("No new data has been processed yet. Please upload and preprocess data first.")

        if 'new_data' in st.session_state:
            st.subheader("Apply Preprocessing to New Data")
            
            preprocessing_params = st.session_state.get('preprocessing_parameters', {})
            if 'loaded_parameters' in st.session_state:
                preprocessing_params = st.session_state.loaded_parameters.get('preprocessing_parameters', {})

            if preprocessing_params:
                st.write("Preprocessing steps that will be applied:")
                if 'spectral_steps' in preprocessing_params:
                    for step in preprocessing_params['spectral_steps']:
                        st.write(f"- {step}")
                if 'wavelet' in preprocessing_params and preprocessing_params['wavelet'].get('applied', False):
                    st.write(f"- Wavelet denoising")
                if 'outlier_removal' in preprocessing_params and preprocessing_params['outlier_removal'].get('applied', False):
                    st.write(f"- Outlier removal (note: not applied to new data)")
                if 'dimensionality_steps' in preprocessing_params and preprocessing_params['dimensionality_steps']:
                    for step in preprocessing_params['dimensionality_steps']:
                        st.write(f"- {step}")
                if 'automated_technique' in preprocessing_params:
                    st.write(f"- Automated {preprocessing_params['automated_technique']} preprocessing")
            else:
                st.warning("No preprocessing parameters found. Data will be used as-is.")

            if st.button("Apply Preprocessing"):
                try:
                    processed_data = st.session_state.new_data.copy()
                    processed_data = processed_data.astype(float)
                    
                    if 'wavelet' in preprocessing_params and preprocessing_params['wavelet'].get('applied', False):
                        st.write("Applying wavelet denoising...")
                        params = preprocessing_params['wavelet']
                        denoiser = WaveletDenoiser(
                            wavelet=params['wavelet'],
                            level=params['level'],
                            threshold_mode=params['threshold_mode']
                        )
                        denoiser.fitted_threshold_ = params['fitted_threshold']
                        processed_data = denoiser.transform(processed_data)

                    trim_steps = [step for step in preprocessing_params.get('spectral_steps', []) if step.startswith('Trim:') or step.startswith('Inverse Trim:')]
                    if trim_steps:
                        st.write("Applying trimming step(s) from training...")
                        
                        temp_file = None
                        try:
                            # Ensure column names are numeric floats for SpectralData
                            trim_data = processed_data.copy()
                            try:
                                trim_data.columns = [float(c) for c in trim_data.columns]
                            except (ValueError, TypeError):
                                pass
                            
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                                temp_file = tmp.name
                                trim_data.to_csv(temp_file, index=False)
                                spectral_trim = SpectralData(temp_file)
                            
                            for trim_step in trim_steps:
                                if trim_step.startswith('Trim:'):
                                    range_part = trim_step.split(': ')[1]
                                    start, end = map(float, range_part.split(' - '))
                                    st.write(f"  Applying Trim: {start} - {end}")
                                    spectral_trim.trim(start=start, end=end)
                                elif trim_step.startswith('Inverse Trim:'):
                                    range_part = trim_step.split(': ')[1]
                                    start, end = map(float, range_part.split(' - '))
                                    st.write(f"  Applying Inverse Trim: {start} - {end}")
                                    spectral_trim.invtrim(start=start, end=end)

                            processed_data = spectral_trim.spc.copy()
                            st.success(f"Trimming applied. Shape: {processed_data.shape}")
                            
                        except Exception as trim_error:
                            st.error(f"Error applying trim steps: {str(trim_error)}")
                        finally:
                            if temp_file and os.path.exists(temp_file):
                                try:
                                    os.unlink(temp_file)
                                except:
                                    pass

                    if 'automated_technique' in preprocessing_params and 'automated_pipeline' in preprocessing_params:
                        st.write(f"Applying automated {preprocessing_params['automated_technique']} preprocessing...")
                        
                        temp_file = None
                        try:
                            # Ensure column names are numeric floats for SpectralData
                            processed_data_for_spectral = processed_data.copy()
                            try:
                                processed_data_for_spectral.columns = [float(c) for c in processed_data_for_spectral.columns]
                            except (ValueError, TypeError):
                                pass  # keep original columns if they can't be converted
                            
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                                temp_file = tmp.name
                                processed_data_for_spectral.to_csv(temp_file, index=False)
                                spectral_new = SpectralData(temp_file)
                            
                            # Force columns to float to prevent str/float comparison errors
                            if hasattr(spectral_new.spc, 'columns'):
                                try:
                                    spectral_new.spc.columns = spectral_new.spc.columns.astype(float)
                                    spectral_new.wav = spectral_new.spc.columns.copy()
                                    spectral_new._wav_raw = spectral_new.spc.columns.copy()
                                except (ValueError, TypeError):
                                    pass
                            
                            for step in preprocessing_params['automated_pipeline']:
                                method = step['method']
                                params = step.get('params', {})
                                
                                st.write(f"  Applying {method}...")
                                
                                try:
                                    if method == 'AsLS':
                                        spectral_new.AsLS(lam=params['lam'], p=params['p'], niter=params['niter'])

                                    elif method == 'Polyfit':
                                        spectral_new.polyfit(order=params['order'], niter=params['niter'])

                                    elif method == 'Pearson':
                                        spectral_new.pearson(u=params['u'], v=params['v'])

                                    elif method == 'Rolling':
                                        spectral_new.rolling(window=params['window'])

                                    elif method == 'Savitzky-Golay':
                                        spectral_new.SGSmooth(window=params['window'], poly=params['poly'])

                                    elif method == 'SNV':
                                        spectral_new.snv()

                                    elif method == 'MSC':
                                        spectral_new.msc()

                                    elif method == 'Detrend':
                                        spectral_new.detrend(order=params['order'])

                                    elif method == 'Area':
                                        spectral_new.area()

                                    elif method == 'Peak Normalization':
                                        spectral_new.peaknorm(wavenumber=params['wave'])

                                    elif method == 'Vector':
                                        spectral_new.vector()

                                    elif method == 'Min-max':
                                        spectral_new.minmax(min_val=params['minv'], max_val=params['maxv'])

                                    elif method == 'Pareto':
                                        spectral_new.pareto()

                                    elif method == 'Mean (spectrum)':
                                        spectral_new.mean_center(option=False)

                                    elif method == 'Mean (wavelength)':
                                        spectral_new.mean_center(option=True)

                                    elif method == 'Last Point':
                                        spectral_new.lastpoint()

                                    elif method == 'Derivative_Subtract':
                                        spectral_new.subtract(spectra=params['subtract_idx'])

                                    elif method == 'Derivative_Reset':
                                        spectral_new.reset()

                                    elif method == 'SG Derivative':
                                        spectral_new.SGDeriv(window=params['window'], poly=params['poly'], order=params['order'])

                                    else:
                                        st.warning(f"Unknown preprocessing method: {method}")
                                        continue
                                        
                                except Exception as method_error:
                                    st.error(f"Error applying {method}: {str(method_error)}")
                                    continue
                            
                            processed_data = spectral_new.spc.copy()
                            st.success("Automated preprocessing applied successfully!")
                            
                        except Exception as pipeline_error:
                            st.error(f"Error applying automated preprocessing pipeline: {str(pipeline_error)}")
                            import traceback
                            st.code(traceback.format_exc())
                        finally:
                            if temp_file and os.path.exists(temp_file):
                                try:
                                    os.unlink(temp_file)
                                except:
                                    pass

                    elif 'spectral_steps' in preprocessing_params and 'spectral_parameters' in preprocessing_params:
                        st.write("Applying manual spectral preprocessing...")
                        st.write(f"Initial data shape: {processed_data.shape}")
                        
                        # Ensure column names are numeric floats for SpectralData
                        manual_data = processed_data.copy()
                        try:
                            manual_data.columns = [float(c) for c in manual_data.columns]
                        except (ValueError, TypeError):
                            pass
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                            manual_data.to_csv(tmp.name, index=False)
                            spectral_new = SpectralData(tmp.name)
                        
                        st.write(f"SpectralData object created with shape: {spectral_new.spc.shape}")
                        
                        spectral_params = preprocessing_params['spectral_parameters']
                        technique_order = []
                        for key in spectral_params.keys():
                            if key.startswith('trim_'):
                                technique_order.append(('Trim', int(key.split('_')[1]), key))
                            elif key.startswith('baseline_'):
                                technique_order.append(('Baseline Correction', int(key.split('_')[1]), key))
                            elif key.startswith('smoothing_'):
                                technique_order.append(('Smoothing', int(key.split('_')[1]), key))
                            elif key.startswith('normalization_'):
                                technique_order.append(('Normalization', int(key.split('_')[1]), key))
                            elif key.startswith('center_'):
                                technique_order.append(('Center', int(key.split('_')[1]), key))
                            elif key.startswith('derivative_'):
                                technique_order.append(('Derivative', int(key.split('_')[-1]), key))
                            elif key.startswith('sg_derivative_'):
                                technique_order.append(('SG Derivative', int(key.split('_')[-1]), key))
                        
                        technique_order.sort(key=lambda x: x[1])
                        
                        for technique, idx, key in technique_order:
                            st.write(f"  Applying {technique} (Step {idx + 1})")
                            params = spectral_params[key]
                            
                            try:
                                if technique == 'Trim':
                                    if params['type'] == "Trim":
                                        spectral_new.trim(start=params['start'], end=params['end'])
                                    else:
                                        spectral_new.invtrim(start=params['start'], end=params['end'])
                                
                                elif technique == 'Baseline Correction':
                                    for method in params['methods']:
                                        if method == "AsLS":
                                            p = params['parameters'][f'AsLS_{idx}']
                                            spectral_new.AsLS(lam=p['lam'], p=p['p'], niter=int(p['niter']))
                                        elif method == "Polyfit":
                                            p = params['parameters'][f'Polyfit_{idx}']
                                            spectral_new.polyfit(order=int(p['order']), niter=int(p['niter']))
                                        elif method == "Pearson":
                                            p = params['parameters'][f'Pearson_{idx}']
                                            spectral_new.pearson(u=int(p['u']), v=int(p['v']))
                                
                                elif technique == 'Smoothing':
                                    for method in params['methods']:
                                        if method == "Rolling":
                                            p = params['parameters'][f'Rolling_{idx}']
                                            spectral_new.rolling(window=int(p['window']))
                                        elif method == "Savitzky-Golay":
                                            p = params['parameters'][f'SG_{idx}']
                                            spectral_new.SGSmooth(window=int(p['window']), poly=int(p['poly']))
                                
                                elif technique == 'Normalization':
                                    for method in params['methods']:
                                        if method == "SNV":
                                            spectral_new.snv()
                                        elif method == "MSC":
                                            spectral_new.msc()
                                        elif method == "Detrend":
                                            p = params['parameters'][f'Detrend_{idx}']
                                            spectral_new.detrend(order=p['order'])
                                        elif method == "Area":
                                            spectral_new.area()
                                        elif method == "Peak Normalization":
                                            p = params['parameters'][f'Peak_{idx}']
                                            spectral_new.peaknorm(wavenumber=p['wave'])
                                        elif method == "Vector":
                                            spectral_new.vector()
                                        elif method == "Min-max":
                                            p = params['parameters'][f'Minmax_{idx}']
                                            spectral_new.minmax(min_val=p['minv'], max_val=p['maxv'])
                                        elif method == "Pareto":
                                            spectral_new.pareto()
                                
                                elif technique == 'Center':
                                    for method in params['methods']:
                                        if method == 'Mean (spectrum)':
                                            spectral_new.mean_center(option=False)
                                        elif method == 'Mean (wavelength)':
                                            spectral_new.mean_center(option=True)
                                        elif method == 'Last Point':
                                            spectral_new.lastpoint()
                                
                                elif technique == 'Derivative':
                                    for option in params['options']:
                                        if option == "Subtract":
                                            spectral_new.subtract(spectra=params['parameters']['subtract_idx'])
                                        elif option == "Reset":
                                            spectral_new.reset()
                                
                                elif technique == 'SG Derivative':
                                    spectral_new.SGDeriv(
                                        window=int(params['window']), 
                                        poly=int(params['poly']), 
                                        order=int(params['order'])
                                    )
                                
                            except Exception as technique_error:
                                st.error(f"    Error applying {technique}: {str(technique_error)}")
                                continue
                        
                        processed_data = spectral_new.spc.copy()
                        processed_data = processed_data.astype(float)
                        
                        if hasattr(spectral_new, 'wav') and len(spectral_new.wav) == processed_data.shape[1]:
                            processed_data.columns = [float(w) for w in spectral_new.wav]
                        
                        st.write(f"Final spectral data shape after all preprocessing: {processed_data.shape}")
                        
                        os.unlink(tmp.name)

                    if 'dimensionality_parameters' in preprocessing_params and 'dimensionality_steps' in preprocessing_params:
                        st.write("Applying dimensionality reduction...")
                        st.write(f"Initial data shape: {processed_data.shape}")
                        
                        # Try to get fitted objects from session state or loaded files
                        fitted_objects = {}
                        if 'loaded_fitted_objects' in st.session_state:
                            fitted_objects = st.session_state.loaded_fitted_objects
                        elif hasattr(st.session_state, 'dim_reducer') and st.session_state.dim_reducer is not None:
                            dim_reducer_obj = st.session_state.dim_reducer
                            fitted_objects['dim_reducer_scaler'] = dim_reducer_obj.scaler
                            fitted_objects['dim_reducer_reducer'] = dim_reducer_obj.reducer
                        elif st.session_state.get('saved_fitted_objects_filename'):
                            fitted_path = st.session_state.saved_fitted_objects_filename
                            if os.path.exists(fitted_path):
                                with open(fitted_path, 'rb') as f:
                                    fitted_objects = pickle.load(f)
                        
                        fitted_scaler = fitted_objects.get('dim_reducer_scaler')
                        fitted_reducer = fitted_objects.get('dim_reducer_reducer')
                        
                        dim_params = preprocessing_params['dimensionality_parameters']
                        technique_order = []
                        
                        for key in dim_params.keys():
                            if key.startswith('scaling_'):
                                technique_order.append(('Scaling', int(key.split('_')[-1]), key))
                            elif key.startswith('pca_'):
                                technique_order.append(('PCA Analysis', int(key.split('_')[-1]), key))
                            elif key.startswith('feature_selection_'):
                                technique_order.append(('Feature Selection', int(key.split('_')[-1]), key))
                        
                        technique_order.sort(key=lambda x: x[1])
                        
                        for technique, idx, key in technique_order:
                            st.write(f"  Applying {technique} (Step {idx + 1})")
                            params = dim_params[key]
                            
                            try:
                                if technique == 'Scaling':
                                    st.write(f"    Scaling method: {params['method']}")
                                    if fitted_scaler is not None:
                                        # Use the fitted scaler from training (transform only)
                                        if hasattr(processed_data, 'values'):
                                            processed_data = pd.DataFrame(
                                                fitted_scaler.transform(processed_data.values),
                                                columns=processed_data.columns
                                            )
                                        else:
                                            processed_data = pd.DataFrame(
                                                fitted_scaler.transform(processed_data)
                                            )
                                        st.success("    Used fitted scaler from training data")
                                    else:
                                        st.warning("    No fitted scaler available — re-fitting on new data (may cause distribution shift)")
                                        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
                                        if params['method'] == 'standard':
                                            scaler = StandardScaler()
                                        elif params['method'] == 'minmax':
                                            scaler = MinMaxScaler()
                                        elif params['method'] == 'robust':
                                            scaler = RobustScaler()
                                        else:
                                            scaler = StandardScaler()
                                        if hasattr(processed_data, 'values'):
                                            processed_data = pd.DataFrame(
                                                scaler.fit_transform(processed_data.values),
                                                columns=processed_data.columns
                                            )
                                        else:
                                            processed_data = pd.DataFrame(
                                                scaler.fit_transform(processed_data)
                                            )
                                    
                                elif technique == 'PCA Analysis':
                                    method = params['method']
                                    pca_params = params['parameters']
                                    st.write(f"    PCA method: {method}")
                                    
                                    if fitted_reducer is not None and hasattr(fitted_reducer, 'transform'):
                                        # Use the fitted PCA from training (transform only)
                                        if hasattr(processed_data, 'values'):
                                            pca_result = fitted_reducer.transform(processed_data.values)
                                        else:
                                            pca_result = fitted_reducer.transform(processed_data)
                                        
                                        n_components = pca_result.shape[1]
                                        processed_data = pd.DataFrame(
                                            pca_result,
                                            columns=[f'PC{i+1}' for i in range(n_components)]
                                        )
                                        st.success(f"    Used fitted PCA from training data → {n_components} components")
                                    else:
                                        st.warning("    No fitted PCA available — re-fitting on new data (may cause distribution shift)")
                                        from sklearn.decomposition import PCA
                                        if method == 'variance':
                                            pca = PCA(n_components=pca_params.get('variance_threshold', 0.95))
                                        elif method == 'fixed':
                                            pca = PCA(n_components=pca_params.get('n_components', 10))
                                        else:
                                            pca = PCA()
                                        
                                        if hasattr(processed_data, 'values'):
                                            pca_result = pca.fit_transform(processed_data.values)
                                        else:
                                            pca_result = pca.fit_transform(processed_data)
                                        
                                        n_components = pca_result.shape[1]
                                        processed_data = pd.DataFrame(
                                            pca_result,
                                            columns=[f'PC{i+1}' for i in range(n_components)]
                                        )
                                    
                                    st.write(f"    Reduced to {n_components} components")
                                
                                elif technique == 'Feature Selection':
                                    selection_method = params['method']
                                    selection_params = params['parameters']
                                    st.write(f"    Feature selection method: {selection_method}")
                                    
                                    original_features = processed_data.shape[1]
                                    
                                    if fitted_reducer is not None and hasattr(fitted_reducer, 'transform'):
                                        # Use the fitted selector from training (transform only)
                                        if hasattr(processed_data, 'values'):
                                            selected_result = fitted_reducer.transform(processed_data.values)
                                        else:
                                            selected_result = fitted_reducer.transform(processed_data)
                                        
                                        processed_data = pd.DataFrame(selected_result)
                                        st.success(f"    Used fitted feature selector from training data")
                                    else:
                                        st.warning("    No fitted feature selector available — re-fitting on new data")
                                        # Fall back to re-fitting (not ideal but better than failing)
                                        from sklearn.feature_selection import VarianceThreshold
                                        if selection_method == 'variance_threshold':
                                            selector = VarianceThreshold(threshold=selection_params.get('threshold', 0.01))
                                            if hasattr(processed_data, 'values'):
                                                selected_result = selector.fit_transform(processed_data.values)
                                            else:
                                                selected_result = selector.fit_transform(processed_data)
                                            processed_data = pd.DataFrame(selected_result)
                                        else:
                                            st.warning(f"    Cannot re-fit {selection_method} without training targets — skipping")
                                            continue
                                    
                                    st.write(f"    Reduced from {original_features} to {processed_data.shape[1]} features")
                            
                            except Exception as technique_error:
                                st.error(f"    Error applying {technique}: {str(technique_error)}")
                                continue
                        
                        processed_data = processed_data.astype(float)
                        
                        if hasattr(processed_data, 'columns'):
                            processed_data.columns = [str(col) for col in processed_data.columns]
                        
                        st.write(f"Final data shape after all dimensionality reduction: {processed_data.shape}")

                    processed_data = processed_data.astype(float)
                    st.session_state.processed_new_data = processed_data
                    st.session_state.preprocessing_applied = True
                    
                    st.success("Preprocessing applied to new data!")
                    st.write(f"Final processed data shape: {processed_data.shape}")
                    
                    if hasattr(processed_data, 'head'):
                        head_data = processed_data.head()
                        
                        clean_data = pd.DataFrame(
                            data=head_data.values.astype(str),
                            columns=[str(col) for col in head_data.columns]
                        )
                        st.dataframe(clean_data)
                    else:    
                        temp_df = pd.DataFrame(processed_data[:10])
                        clean_data = pd.DataFrame(
                            data=temp_df.values.astype(str),
                            columns=[str(col) for col in temp_df.columns]
                        )
                        st.dataframe(clean_data)

                except Exception as e:
                    st.error(f"Error applying preprocessing: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

        col1, col2 = st.columns(2)
        processed = st.session_state.get('processed_new_data', None)
        if processed is not None and (processed.size > 0 if hasattr(processed, 'size') else len(processed) > 0) and st.session_state.get('preprocessing_applied', False):
            with col1:
                if st.button("Proceed to Prediction"):
                    st.session_state.step = 14
                    st.rerun()
        else:
            with col1:
                st.info("Apply preprocessing first to enable prediction")

        with col2:
            if st.session_state.direct_prediction_mode:
                if st.button("Start Over"):
                    keys_to_clear = ['loaded_model', 'loaded_parameters', 'loaded_fitted_objects', 'new_data', 'processed_new_data']
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
            else:
                if st.button("Go Back to Model Saving"):
                    st.session_state.step = 12
                    st.rerun()

#####################################################################################################################################

    elif st.session_state.step == 14:
        st.header("Step 14: Prediction")
        
        if st.session_state.direct_prediction_mode and 'loaded_model' not in st.session_state:
            st.info("Direct Prediction Mode - Load your saved model and prediction data")
            
            st.subheader("Load Saved Model and Parameters")
            col1, col2 = st.columns(2)
            
            with col1:
                uploaded_model = st.file_uploader("Upload saved model file (.pkl)", type=["pkl"], key="prediction_model")
            with col2:
                uploaded_params = st.file_uploader("Upload parameters JSON file", type=["json"], key="prediction_params")
            
            if uploaded_model and uploaded_params:
                try:
                    st.session_state.loaded_model = pickle.load(uploaded_model)
                    params_data = json.load(uploaded_params)
                    st.session_state.loaded_parameters = params_data
                    
                    st.success("Model and parameters loaded successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error loading model/parameters: {str(e)}")
                    return
        
        if st.session_state.direct_prediction_mode and 'processed_new_data' not in st.session_state:
            st.subheader("Load Preprocessed Data for Prediction")
            st.info("Upload data that has already been preprocessed using the same steps as your training data")
            
            prediction_data_file = st.file_uploader("Upload preprocessed data for prediction", type=["csv", "xlsx", "txt"])
            
            if prediction_data_file:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{prediction_data_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(prediction_data_file.getbuffer())
                        file_path = tmp_file.name

                    prediction_data = rd.read_data(file_path)
                    st.session_state.processed_new_data = prediction_data.copy()
                    st.session_state.preprocessing_applied = True

                    st.success("Preprocessed data loaded successfully!")
                    st.write(f"Data shape: {prediction_data.shape}")
                    st.dataframe(prediction_data.head())

                    os.unlink(file_path)

                except Exception as e:
                    st.error(f"Error loading prediction data: {str(e)}")
                    return
        
        if 'processed_new_data' not in st.session_state:
            st.error("Please process new data first or upload preprocessed data")
            if st.button("Go back to New Data Processing"):
                st.session_state.step = 13
                st.rerun()
            return
        
        model_to_use = None
        model_info = {}
        
        if st.session_state.direct_prediction_mode and 'loaded_model' in st.session_state:
            model_to_use = st.session_state.loaded_model
            model_info = st.session_state.loaded_parameters.get('model_info', {})
            model_params = st.session_state.loaded_parameters.get('model_parameters', {})
            model_info.update(model_params)
            st.success(f"Using loaded model: {model_info.get('model_name', 'Unknown')}")
        elif not st.session_state.direct_prediction_mode and 'trained_model' in st.session_state:
            model_to_use = st.session_state.trained_model
            model_info = st.session_state.model_parameters
            st.success(f"Using trained model: {model_info.get('model_name', 'Unknown')}")
        
        if model_to_use is None:
            st.error("No model available for prediction")
            return
        
        st.subheader("Make Predictions")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Model Information")
            st.write(f"Model: {model_info.get('model_name', 'Unknown')}")
            st.write(f"Type: {model_info.get('model_type', 'Unknown')}")
            st.write(f"Method: {model_info.get('training_method', 'Unknown')}")
            
            if st.session_state.direct_prediction_mode and 'loaded_parameters' in st.session_state:
                user_info = st.session_state.loaded_parameters.get('user_info', {})
                if user_info.get('user_name'):
                    st.write(f"Created by: {user_info['user_name']}")
                if user_info.get('creation_date'):
                    creation_date = user_info['creation_date'][:10]
                    st.write(f"Created: {creation_date}")
            
            if 'performance_metrics' in model_info:
                st.write("Performance Metrics:")
                for metric, value in model_info['performance_metrics'].items():
                    if isinstance(value, (int, float)):
                        st.write(f"  - {metric}: {value:.4f}")
        
        with col2:
            st.subheader("Data Information")
            new_data = st.session_state.processed_new_data.copy()
            st.write(f"New Data Shape: {new_data.shape}")
            st.write(f"New Data Samples: {new_data.shape[0]}")
            st.write(f"New Data Features: {new_data.shape[1]}")
            
            expected_features = 0
            if st.session_state.direct_prediction_mode and 'loaded_parameters' in st.session_state:
                feature_info = st.session_state.loaded_parameters.get('data_info', {})
                expected_features = len(feature_info.get('feature_columns', []))
            elif 'X_train' in st.session_state:
                expected_features = st.session_state.X_train.shape[1]
            
            if expected_features > 0:
                st.write(f"Expected Features: {expected_features}")
                if new_data.shape[1] != expected_features:
                    st.warning(f"Feature count mismatch! Expected {expected_features}, got {new_data.shape[1]}")
                else:
                    st.success("Feature count matches")
        
        if st.button("Make Predictions", type="primary", use_container_width=True):
            try:
                with st.spinner("Making predictions..."):
                    new_data = st.session_state.processed_new_data.copy()
                    
                    if hasattr(new_data, 'values'):
                        prediction_data = new_data.values
                    else:
                        prediction_data = np.array(new_data)
                    
                    prediction_data = prediction_data.astype(float)
                    
                    if np.any(np.isnan(prediction_data)) or np.any(np.isinf(prediction_data)):
                        st.error("Data contains NaN or infinite values. Please check preprocessing.")
                        return
                    
                    is_two_part_model = (
                        isinstance(model_to_use, dict) and 
                        'classification_model' in model_to_use and 
                        'regression_model' in model_to_use
                    )
                    
                    if is_two_part_model:
                        clf_model = model_to_use['classification_model']
                        reg_model = model_to_use['regression_model']
                        
                        binary_predictions = clf_model.predict(prediction_data)
                        
                        predictions = np.zeros(len(prediction_data), dtype=float)
                        
                        nonzero_mask = binary_predictions == 1
                        if np.any(nonzero_mask):
                            X_nonzero = prediction_data[nonzero_mask]
                            reg_predictions = reg_model.predict(X_nonzero)
                            predictions[nonzero_mask] = reg_predictions
                        
                        st.info(f"Two-part model: {np.sum(binary_predictions == 0)} samples predicted as zero, {np.sum(binary_predictions == 1)} samples predicted as non-zero")
                    else:
                        predictions = model_to_use.predict(prediction_data)
                    
                    if model_info.get('model_type') == 'regression' or model_info.get('model_type') == 'zero_inflated':
                        original_negative_count = np.sum(predictions < 0)
                        if original_negative_count > 0:
                            st.warning(f"Found {original_negative_count} negative predictions. Replacing with 0.")
                            predictions = np.maximum(predictions, 0)
                    
                    st.session_state.predictions = predictions
                    
                st.success("Predictions made successfully!")
                
                st.subheader("Prediction Results")
                
                if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                    results_df = pd.DataFrame(predictions)
                    results_df.insert(0, 'Sample_Index', range(len(predictions)))
                    
                    target_columns = None
                    if st.session_state.direct_prediction_mode and 'loaded_parameters' in st.session_state:
                        target_info = st.session_state.loaded_parameters.get('data_info', {})
                        target_columns = target_info.get('target_columns', [])
                    elif 'target_columns' in st.session_state:
                        target_columns = st.session_state.target_columns
                    
                    if target_columns and len(target_columns) == predictions.shape[1]:
                        results_df.columns = ['Sample_Index'] + target_columns
                    else:
                        results_df.columns = ['Sample_Index'] + [f'Target_{i+1}' for i in range(predictions.shape[1])]
                else:
                    predictions_flat = predictions.flatten() if len(predictions.shape) > 1 else predictions
                    results_df = pd.DataFrame({
                        'Sample_Index': range(len(predictions_flat)),
                        'Prediction': predictions_flat
                    })
                    
                    target_name = 'Prediction'
                    if st.session_state.direct_prediction_mode and 'loaded_parameters' in st.session_state:
                        target_info = st.session_state.loaded_parameters.get('data_info', {})
                        target_columns = target_info.get('target_columns', [])
                        if target_columns and len(target_columns) == 1:
                            target_name = target_columns[0]
                    elif 'target_columns' in st.session_state and len(st.session_state.target_columns) == 1:
                        target_name = st.session_state.target_columns[0]
                    
                    results_df.columns = ['Sample_Index', target_name]
                
                st.dataframe(results_df, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                pred_values = predictions.flatten() if len(predictions.shape) > 1 else predictions
                
                with col1:
                    st.metric("Total Predictions", len(pred_values))
                with col2:
                    st.metric("Mean Prediction", f"{pred_values.mean():.4f}")
                with col3:
                    st.metric("Std Deviation", f"{pred_values.std():.4f}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Min Prediction", f"{pred_values.min():.4f}")
                with col2:
                    st.metric("Max Prediction", f"{pred_values.max():.4f}")
                with col3:
                    st.metric("Median Prediction", f"{np.median(pred_values):.4f}")
                
                if is_two_part_model:
                    st.subheader("Two-Part Model Details")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Predicted as Zero", int(np.sum(predictions == 0)))
                    with col2:
                        st.metric("Predicted as Non-Zero", int(np.sum(predictions > 0)))
                
                st.subheader("Prediction Visualization")
                
                try:
                    if results_df.shape[1] == 2:
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.plot(results_df['Sample_Index'], results_df.iloc[:, 1], 'o-', alpha=0.7, linewidth=2, markersize=4)
                        ax.set_xlabel('Sample Index')
                        ax.set_ylabel('Predicted Value')
                        ax.set_title('Predictions on New Data')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        plt.close(fig)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.hist(results_df.iloc[:, 1], bins=min(30, len(results_df)//5), alpha=0.7, edgecolor='black')
                        ax.set_xlabel('Predicted Value')
                        ax.set_ylabel('Frequency')
                        ax.set_title('Distribution of Predictions')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        plt.close(fig)
                        
                    else:
                        n_targets = min(3, results_df.shape[1]-1)
                        fig, axes = plt.subplots(1, n_targets, figsize=(5*n_targets, 5))
                        if n_targets == 1:
                            axes = [axes]
                        
                        for i, col in enumerate(results_df.columns[1:n_targets+1]):
                            ax = axes[i]
                            ax.plot(results_df['Sample_Index'], results_df[col], 'o-', alpha=0.7, linewidth=2, markersize=4)
                            ax.set_xlabel('Sample Index')
                            ax.set_ylabel(f'Predicted {col}')
                            ax.set_title(f'Predictions: {col}')
                            ax.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                        
                except Exception as plot_error:
                    st.warning(f"Could not create prediction plots: {plot_error}")
                
                st.subheader("Download Results")
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                if st.session_state.user_name:
                    filename = f"predictions_{st.session_state.user_name}_{timestamp}.csv"
                else:
                    filename = f"predictions_{timestamp}.csv"
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "Download Predictions as CSV",
                    data=csv,
                    file_name=filename,
                    mime="text/csv",
                    use_container_width=True
                )
                
                with st.expander("Detailed Prediction Summary"):
                    st.write("Prediction Details:")
                    st.write(f"- Number of samples: {len(predictions)}")
                    st.write(f"- Prediction shape: {predictions.shape}")
                    st.write(f"- Data type: {predictions.dtype}")
                    
                    if is_two_part_model:
                        st.write(f"- Model type: Two-part (Zero-Inflated)")
                        st.write(f"- Classification model: {model_to_use.get('classification_name', 'Unknown')}")
                        st.write(f"- Regression model: {model_to_use.get('regression_name', 'Unknown')}")
                    
                    if len(predictions.shape) == 1 or predictions.shape[1] == 1:
                        st.write(f"- Min prediction: {pred_values.min():.4f}")
                        st.write(f"- Max prediction: {pred_values.max():.4f}")
                        st.write(f"- Median prediction: {np.median(pred_values):.4f}")
                        st.write(f"- 25th percentile: {np.percentile(pred_values, 25):.4f}")
                        st.write(f"- 75th percentile: {np.percentile(pred_values, 75):.4f}")
                    
                    st.write("First 10 predictions:")
                    st.dataframe(results_df.head(10))
                    
                    st.write("Last 10 predictions:")
                    st.dataframe(results_df.tail(10))
                    
            except Exception as e:
                st.error(f"Error making predictions: {str(e)}")
                st.error("Please check that your new data has the same preprocessing applied as the training data")
                
                with st.expander("Debug Information"):
                    st.write("New Data Info:")
                    st.write(f"- Shape: {new_data.shape}")
                    st.write(f"- Type: {type(new_data)}")
                    if hasattr(new_data, 'columns'):
                        st.write(f"- Columns (first 10): {list(new_data.columns[:10])}")
                    if hasattr(new_data, 'dtypes'):
                        st.write(f"- Data types: {new_data.dtypes.value_counts().to_dict()}")
                    
                    st.write("Model Info:")
                    st.write(f"- Model type: {type(model_to_use)}")
                    st.write(f"- Is two-part model: {isinstance(model_to_use, dict) and 'classification_model' in model_to_use}")
                    
                    if hasattr(new_data, 'values'):
                        data_values = new_data.values
                        st.write(f"- Contains NaN: {np.any(np.isnan(data_values))}")
                        st.write(f"- Contains Inf: {np.any(np.isinf(data_values))}")
                        st.write(f"- Data range: {data_values.min():.4f} to {data_values.max():.4f}")
                    
                    import traceback
                    st.code(traceback.format_exc())
                    
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Process More Data", use_container_width=True):
                st.session_state.step = 13
                keys_to_clear = ['processed_new_data', 'new_data', 'preprocessing_applied', 'predictions']
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        
        with col2:
            if st.button("Start New Pipeline", use_container_width=True):
                for key in list(st.session_state.keys()):
                    if key != 'user_name':
                        del st.session_state[key]
                st.session_state.step = 1
                st.session_state.direct_prediction_mode = False
                st.rerun()
        
        with col3:
            if st.button("Back to Data Processing", use_container_width=True):
                st.session_state.step = 13
                st.rerun()

    # Sidebar Navigation and Status
    st.sidebar.markdown("---")
    st.sidebar.subheader("Quick Navigation")
    
    if st.session_state.direct_prediction_mode:
        st.sidebar.info("Direct Prediction Mode")
    else:
        st.sidebar.info("Full Pipeline Mode")
    
    if st.session_state.user_name:
        st.sidebar.markdown("---")
        st.sidebar.subheader(f"{st.session_state.user_name}'s Options")
        
        if st.sidebar.button("My Models", help="Go to prediction with my saved models"):
            st.session_state.direct_prediction_mode = True
            st.session_state.step = 13
            st.rerun()
        
        if st.sidebar.button("New Analysis", help="Start fresh pipeline"):
            st.session_state.direct_prediction_mode = False
            st.session_state.step = 1
            st.rerun()
    
    # Step navigation buttons
    for i, step_name in enumerate(progress_steps, 1):
        if st.sidebar.button(f"Step {i}: {step_name}"):
            if i == 13 or i == 14:  # Direct prediction steps
                st.session_state.step = i
                st.rerun()
            elif i == 1 or i == 2:
                st.session_state.direct_prediction_mode = False
                st.session_state.step = i
                st.rerun()
            else:
                # Check prerequisites
                can_navigate = True
                error_msg = ""
                
                if not st.session_state.get('data_loaded', False) and i > 1:
                    can_navigate = False
                    error_msg = "Please upload data first!"
                elif not st.session_state.get('targets_set', False) and i > 2:
                    can_navigate = False
                    error_msg = "Please set targets first!"
                elif not st.session_state.get('data_split', False) and i > 4:
                    can_navigate = False
                    error_msg = "Please complete train-test split first!"
                
                if can_navigate:
                    st.session_state.direct_prediction_mode = False
                    st.session_state.step = i
                    # Clear downstream states when jumping to earlier steps
                    if i < st.session_state.get('step', 1):
                        reset_to_step(i)
                    st.rerun()
                else:
                    st.sidebar.error(error_msg)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Current Session Status")
    if st.session_state.user_name:
        st.sidebar.success(f"User: {st.session_state.user_name}")
    
    if st.session_state.get('data_loaded', False):
        st.sidebar.success("Data Loaded")
    if st.session_state.get('targets_set', False):
        st.sidebar.success("Targets Set")
    if st.session_state.get('data_split', False):
        st.sidebar.success("Data Split Created")
    if st.session_state.model_trained:
        st.sidebar.success("Model Trained")
    if st.session_state.model_saved:
        st.sidebar.success("Model Saved")
    if 'loaded_model' in st.session_state:
        st.sidebar.success("Model Loaded")
    if 'predictions' in st.session_state:
        st.sidebar.success("Predictions Made")

    # Show skipped steps
    if st.session_state.skipped_steps:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Skipped Steps Summary")
        for skip_step in sorted(st.session_state.skipped_steps):
            if skip_step <= len(progress_steps):
                st.sidebar.warning(f"{progress_steps[skip_step-1]}")
        
        if st.sidebar.button("Clear All Skips"):
            st.session_state.skipped_steps.clear()
            st.sidebar.success("All skips cleared!")
            st.rerun()
    
    # Clean up temporary files
    for temp_file_attr in ['temp_file', 'temp_file_train', 'temp_file_test']:
        if hasattr(st.session_state, temp_file_attr):
            temp_file = getattr(st.session_state, temp_file_attr)
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass


main()

from chatbot import render_chatbot
render_chatbot("05_Full_Pipeline")

