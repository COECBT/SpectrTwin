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
from spectra_specific.RamanSpectra import RamanPreprocessingOptimizer
from spectra_specific.FTIRSpectra import FTIRPreprocessingOptimizer
from FFT import FFTProcessor
import copy 
from copy import deepcopy
from mpls import MultiWayPLS

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
                if st.button("Skip to Mpls"):
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
        st.header("Step 4: MPLS - Multiway PLS for Batch Process Monitoring")
        
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
                        st.download_button("Download Training Stats CSV",
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
            if st.button(" Back to Train/Test Split "):
                st.session_state.step = 3
                st.rerun()
        
        with col2:
            if st.session_state.mpls_model is not None:
                if st.button("Proceed to  ", type="primary"):
                    st.session_state.step = 10
                    st.rerun()
                    
        with col3:
            if st.button("Skip MPLS"):
                st.session_state.step = 9
                st.info("Skipped MPLS analysis")
                st.rerun()
                    
                    
app()

from chatbot import render_chatbot
render_chatbot("04_Control chart")

