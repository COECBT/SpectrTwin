import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import json
import pickle
from datetime import datetime
import re
import copy
import contextlib
import io
from preprocess import SpectralData
from midel import WaveletDenoiser, OutlierRemover         
from spectra_specific.NIRSpectra import NIRPreprocessingOptimizer
from spectra_specific.RamanSpectra import RamanPreprocessingOptimizer
from spectra_specific.RamanSpectra1 import RamanPreprocessingOptimizer as RamanPureOptimizer
from spectra_specific.FTIRSpectra import FTIRPreprocessingOptimizer
from spectra_specific.Mass_spectra import MassSpectralPreprocessingOptimizer
from FFT import FFTProcessor
from opls import OPLS
import base64

@contextlib.contextmanager
def _suppress_stdout():
    with contextlib.redirect_stdout(io.StringIO()):
        yield

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    else:
        return obj

def save_parameters_to_json(params, filename):
    params_clean = convert_numpy_types(params)
    with open(filename, 'w') as f:
        json.dump(params_clean, f, indent=4, ensure_ascii=False)
    return filename

def create_download_link(data, filename, file_label):
    if isinstance(data, pd.DataFrame):
        csv = data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{file_label}</a>'
    else:
        with open(data, 'rb') as f:
            bytes_data = f.read()
        b64 = base64.b64encode(bytes_data).decode()
        ext = filename.split('.')[-1]
        href = f'<a href="data:file/{ext};base64,{b64}" download="{filename}">{file_label}</a>'
    return href

def plot_spectra_comparison(original_data, processed_data, original_wav, processed_wav, title="Spectra Comparison"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    n_plot = min(10, len(original_data))
    for i in range(n_plot):
        ax1.plot(original_wav[:original_data.shape[1]], original_data.iloc[i].values, alpha=0.6)
    ax1.set_title("Original Spectra")
    ax1.set_xlabel("Wavelength")
    ax1.set_ylabel("Intensity")
    ax1.grid(True, alpha=0.3)

    for i in range(n_plot):
        ax2.plot(processed_wav[:processed_data.shape[1]], processed_data.iloc[i].values, alpha=0.6)
    ax2.set_title("Preprocessed Spectra")
    ax2.set_xlabel("Wavelength")
    ax2.set_ylabel("Intensity")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def main():
    st.title("Spectroscopic Data Preprocessing Pipeline")

    if 'preprocessing_params' not in st.session_state:
        st.session_state.preprocessing_params = {}
    if 'preprocessing_history' not in st.session_state:
        st.session_state.preprocessing_history = []
    if 'user_name' not in st.session_state:
        st.session_state.user_name = ""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'target_set' not in st.session_state:
        st.session_state.target_set = False

    st.sidebar.title("Pipeline Status")

    steps = {
        1: "Data Upload",
        2: "Target Selection",
        3: "Outlier Removal",
        4: "Wavelet Denoising",
        5: "Standard Preprocessing",
        6: "Advanced Preprocessing",
        7: "Export Results"
    }

    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1

    for step_num, step_name in steps.items():
        if st.session_state.current_step > step_num:
            st.sidebar.success(f"Step {step_num}: {step_name}")
        elif st.session_state.current_step == step_num:
            st.sidebar.info(f"Step {step_num}: {step_name}")
        else:
            st.sidebar.write(f"Step {step_num}: {step_name}")

    if not st.session_state.user_name:
        st.header("User Information")
        user_name = st.text_input("Enter Your Name:", placeholder="Enter your name")
        if user_name:
            st.session_state.user_name = user_name
            st.rerun()
        else:
            st.info("Please enter your name to proceed.")
            return
    else:
        st.success(f"Welcome, {st.session_state.user_name}")

    if st.session_state.current_step == 1:
        st.header("Step 1: Data Upload")

        uploaded_file = st.file_uploader("Upload CSV, XLSX or TXT file", type=["csv", "xlsx", "txt"])

        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    data = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.txt'):
                    data = pd.read_csv(uploaded_file, delimiter='\t')

                st.session_state.original_data = data.copy()
                st.session_state.current_data = data.copy()
                st.session_state.data_loaded = True

                st.success("Data loaded successfully")
                st.dataframe(data.head())
                st.info(f"Data shape: {data.shape[0]} samples, {data.shape[1]} features")

                if st.button("Proceed to Target Selection"):
                    st.session_state.current_step = 2
                    st.rerun()

            except Exception as e:
                st.error(f"Error loading data: {str(e)}")

    elif st.session_state.current_step == 2:
        st.header("Step 2: Target Selection")

        if not st.session_state.data_loaded:
            st.error("Please upload data first")
            if st.button("Go back to Data Upload"):
                st.session_state.current_step = 1
                st.rerun()
            return

        data = st.session_state.current_data.copy()

        st.subheader("Optional: Drop Columns")
        drop_columns = st.multiselect("Select columns to drop (optional)", data.columns)

        if drop_columns and st.button("Drop Selected Columns"):
            data = data.drop(columns=drop_columns)
            st.session_state.current_data = data.copy()
            st.success(f"Dropped columns: {', '.join(drop_columns)}")
            st.rerun()

        st.dataframe(data.head())

        target_columns = st.multiselect("Select Target Columns", data.columns)

        if st.button("Set Targets"):
            if target_columns:
                st.session_state.target_columns = target_columns
                X = data.drop(columns=target_columns)
                y = data[target_columns]

                st.session_state.X = X
                st.session_state.y = y
                st.session_state.target_set = True

                try:
                    x_axis = X.columns.astype(float)
                    st.session_state.x_axis = x_axis
                except ValueError:
                    st.session_state.x_axis = np.arange(X.shape[1])

                st.success(f"Target columns set: {target_columns}")
                st.info(f"Features: {X.shape}, Targets: {y.shape}")
            else:
                st.error("Please select at least one target column")

        if st.session_state.target_set:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Proceed to Outlier Removal"):
                    st.session_state.current_step = 3
                    st.rerun()
            with col2:
                if st.button("Skip to Wavelet Denoising"):
                    st.session_state.current_step = 4
                    st.rerun()

    elif st.session_state.current_step == 3:
        st.header("Step 3: Outlier Removal")

        if not st.session_state.target_set:
            st.error("Please complete target selection first")
            return

        X = st.session_state.X
        y = st.session_state.y
        x_axis = st.session_state.x_axis

        apply_outlier_removal = st.checkbox("Apply outlier removal based on standard deviation")

        if apply_outlier_removal:
            threshold = st.slider("Outlier Detection Threshold (sigma)", 1.0, 5.0, 3.0, 0.1)

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

                    st.session_state.X = filtered_X
                    st.session_state.y = filtered_y

                    st.session_state.preprocessing_params['outlier_removal'] = {
                        'threshold': float(threshold),
                        'applied': True,
                        'removed_samples': X.shape[0] - filtered_X.shape[0]
                    }
                    st.session_state.preprocessing_history.append(f"Outlier Removal: threshold={threshold}")

                    st.success(f"Removed {X.shape[0] - filtered_X.shape[0]} outliers")
                    st.info(f"Remaining samples: {filtered_X.shape[0]}")

                    fig, ax = plt.subplots(figsize=(12, 6))
                    for i in range(min(50, filtered_X.shape[0])):
                        ax.plot(x_axis[:filtered_X.shape[1]], filtered_X.iloc[i].values, alpha=0.5)
                    ax.set_xlabel("Wavenumber")
                    ax.set_ylabel("Intensity")
                    ax.set_title("Spectra After Outlier Removal")
                    ax.grid(True)
                    st.pyplot(fig)
                    plt.close(fig)

                except Exception as e:
                    st.error(f"Error in outlier removal: {str(e)}")
        else:
            st.session_state.preprocessing_params['outlier_removal'] = {'applied': False}
            st.info("Outlier removal skipped")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Proceed to Wavelet Denoising"):
                st.session_state.current_step = 4
                st.rerun()
        with col2:
            if st.button("Skip to Standard Preprocessing"):
                st.session_state.current_step = 5
                st.rerun()

    elif st.session_state.current_step == 4:
        st.header("Step 4: Wavelet Denoising")

        if not st.session_state.target_set:
            st.error("Please complete previous steps first")
            return

        X = st.session_state.X
        y = st.session_state.y
        x_axis = st.session_state.x_axis

        st.subheader("Original Spectra")
        fig, ax = plt.subplots(figsize=(12, 6))
        for i in range(min(50, X.shape[0])):
            ax.plot(x_axis[:X.shape[1]], X.iloc[i].values, alpha=0.5)
        ax.set_xlabel("Wavenumber")
        ax.set_ylabel("Intensity")
        ax.set_title("Original Spectra")
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

                    with st.spinner("Applying wavelet denoising..."):
                        denoiser.fit(X)
                        denoised_X = denoiser.transform(X)

                    st.session_state.X = denoised_X
                    st.session_state.fitted_denoiser = denoiser
                    st.session_state.preprocessing_params['wavelet'] = {
                        'wavelet': str(wavelet),
                        'level': int(level),
                        'threshold_mode': str(mode),
                        'fitted_threshold': float(denoiser.fitted_threshold_),
                        'applied': True
                    }
                    st.session_state.preprocessing_history.append(f"Wavelet Denoising: {wavelet}, level={level}, mode={mode}")

                    st.subheader("Denoised Spectra")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    for i in range(min(50, denoised_X.shape[0])):
                        ax.plot(x_axis[:denoised_X.shape[1]], denoised_X.iloc[i].values, alpha=0.5)
                    ax.set_xlabel("Wavenumber")
                    ax.set_ylabel("Intensity")
                    ax.set_title("Wavelet Denoised Spectra")
                    ax.grid(True)
                    st.pyplot(fig)
                    plt.close(fig)

                    st.success(f"Wavelet denoising completed. Shape: {denoised_X.shape}")
                    st.info(f"Fitted threshold: {denoiser.fitted_threshold_:.6f}")

                except Exception as e:
                    st.error(f"Error in wavelet denoising: {str(e)}")

        with col2:
            if st.button("Skip Wavelet Denoising"):
                st.session_state.preprocessing_params['wavelet'] = {'applied': False}
                st.info("Wavelet denoising skipped")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Proceed to Standard Preprocessing"):
                st.session_state.current_step = 5
                st.rerun()
        with col2:
            if st.button("Skip to Advanced Preprocessing"):
                st.session_state.current_step = 6
                st.rerun()

    elif st.session_state.current_step == 5:
        st.header("Step 5: Standard Preprocessing")

        if not st.session_state.target_set:
            st.error("Please complete previous steps first")
            return

        X = st.session_state.X
        y = st.session_state.y

        if 'spectral_data' not in st.session_state or st.session_state.spectral_data.spc.shape[0] != st.session_state.X.shape[0] or st.button("Reset Preprocessing"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                X.to_csv(tmp.name, index=False)
                st.session_state.spectral_data = SpectralData(tmp.name)
                st.session_state.temp_file = tmp.name

            st.session_state.standard_preprocessing_done = False
            st.success("Preprocessing initialized")
            st.rerun()

        spectral_data = st.session_state.spectral_data

        st.subheader("Original Spectra")
        fig, ax = plt.subplots(figsize=(10, 5))
        for i in range(min(100, len(spectral_data.spc))):
            ax.plot(spectral_data.wav[:spectral_data.spc.shape[1]], spectral_data.spc.iloc[i], alpha=0.7)
        ax.set_title("Original Spectra")
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Intensity")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

        technique = st.selectbox(
            "Choose your analytical technique:",
            ["Raman Spectroscopy", "NIR Spectroscopy", "FTIR Spectroscopy", "Mass Spectrometry"]
        )

        preprocessing_mode = st.radio("Preprocessing Mode", ["Manual", "Automated"], horizontal=True)

        if preprocessing_mode == "Manual":
            st.subheader("Manual Preprocessing Configuration")

            mandatory_steps = []
            if technique == "Raman Spectroscopy":
                mandatory_steps = ['Baseline Correction', 'Smoothing']
            elif technique == "NIR Spectroscopy":
                mandatory_steps = ['Normalization', 'SG Derivative']
            elif technique == "FTIR Spectroscopy":
                mandatory_steps = ['Baseline Correction', 'Normalization']

            if mandatory_steps:
                st.info(f"Necessary steps enforced for {technique}: {', '.join(mandatory_steps)}")

            all_steps = ['Trim', 'Baseline Correction', 'Smoothing', 'Normalization', 'Center', 'Derivative', 'SG Derivative']
            optional_choices = [s for s in all_steps if s not in mandatory_steps]

            selected_optional = st.multiselect("Choose optional preprocessing techniques:", optional_choices)

            selected_techniques = [s for s in all_steps if s in mandatory_steps or s in selected_optional]

            current_params = {}

            for idx, step_name in enumerate(selected_techniques):
                st.markdown(f"### {step_name} (Step {idx + 1})")

                if step_name == 'Trim':
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        trim_type = st.radio(f"Trim Type (Step {idx + 1}):", ["Trim", "Inverse Trim"], key=f"trim_type_{idx}")
                    with col2:
                        start = st.number_input(f"Start Wavelength (Step {idx + 1})",
                                            value=float(spectral_data.wav.min()),
                                            key=f"trim_start_{idx}")
                    with col3:
                        end = st.number_input(f"End Wavelength (Step {idx + 1})",
                                            value=float(spectral_data.wav.max()),
                                            key=f"trim_end_{idx}")

                    current_params[f'trim_{idx}'] = {
                        'type': trim_type,
                        'start': start,
                        'end': end
                    }

                elif step_name == 'Baseline Correction':
                    default_method = ["AsLS"] if step_name in mandatory_steps else []
                    baseline_methods = st.multiselect(f"Baseline Correction Methods (Step {idx + 1}):",
                                                    ["AsLS", "Polyfit", "Pearson"],
                                                    default=default_method,
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

                elif step_name == 'Smoothing':
                    default_method = ["Savitzky-Golay"] if step_name in mandatory_steps else []
                    smoothing_methods = st.multiselect(f"Smoothing Methods (Step {idx + 1}):",
                                                     ["Rolling", "Savitzky-Golay"],
                                                     default=default_method,
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

                elif step_name == 'Normalization':
                    default_method = ["SNV"] if step_name in mandatory_steps else []
                    normalization_methods = st.multiselect(f"Normalization Methods (Step {idx + 1}):",
                                                         ["SNV", "MSC", "Detrend", "Area", "Peak Normalization", "Vector", "Min-max", "Pareto"],
                                                         default=default_method,
                                                         key=f"normalization_methods_{idx}")

                    normalization_params = {}
                    for method in normalization_methods:
                        if method == "Detrend":
                            order = st.number_input(f"Detrend Order (Step {idx + 1})",
                                                  value=2, key=f"detrend_order_{idx}")
                            normalization_params[f'Detrend_{idx}'] = {'order': order}
                        elif method == "Peak Normalization":
                            wave = st.number_input(f"Peak Wavenumber (Step {idx + 1})",
                                                 value=float(spectral_data.wav.median()),
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

                elif step_name == 'Center':
                    center_methods = st.multiselect(f"Centering Methods (Step {idx + 1}):",
                                                  ["Mean (spectrum)", "Mean (wavelength)", "Last Point"],
                                                  key=f"center_methods_{idx}")

                    current_params[f'center_{idx}'] = {
                        'methods': center_methods
                    }

                elif step_name == 'Derivative':
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

                elif step_name == 'SG Derivative':
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
                    st.session_state.preprocessing_params['spectral_parameters'] = current_params

                    applied_steps = []

                    for idx, step_name in enumerate(selected_techniques):
                        st.write(f"Applying {step_name} (Step {idx + 1})...")

                        if step_name == 'Trim':
                            params = current_params[f'trim_{idx}']
                            if params['type'] == "Trim":
                                spectral_data.trim(start=params['start'], end=params['end'])
                                step_name_log = f"Trim: {params['start']:.2f} - {params['end']:.2f}"
                            else:
                                spectral_data.invtrim(start=params['start'], end=params['end'])
                                step_name_log = f"Inverse Trim: {params['start']:.2f} - {params['end']:.2f}"
                            applied_steps.append(step_name_log)

                        elif step_name == 'Baseline Correction':
                            params = current_params[f'baseline_{idx}']
                            for method in params['methods']:
                                if method == "AsLS":
                                    p = params['parameters'][f'AsLS_{idx}']
                                    with _suppress_stdout():
                                        spectral_data.AsLS(lam=p['lam'], p=p['p'], niter=int(p['niter']))
                                elif method == "Polyfit":
                                    p = params['parameters'][f'Polyfit_{idx}']
                                    with _suppress_stdout():
                                        spectral_data.polyfit(order=int(p['order']), niter=int(p['niter']))
                                elif method == "Pearson":
                                    p = params['parameters'][f'Pearson_{idx}']
                                    with _suppress_stdout():
                                        spectral_data.pearson(u=int(p['u']), v=int(p['v']))
                                applied_steps.append(f"Baseline Correction: {method}")

                        elif step_name == 'Smoothing':
                            params = current_params[f'smoothing_{idx}']
                            for method in params['methods']:
                                if method == "Rolling":
                                    p = params['parameters'][f'Rolling_{idx}']
                                    spectral_data.rolling(window=int(p['window']))
                                elif method == "Savitzky-Golay":
                                    p = params['parameters'][f'SG_{idx}']
                                    spectral_data.SGSmooth(window=int(p['window']), poly=int(p['poly']))
                                applied_steps.append(f"Smoothing: {method}")

                        elif step_name == 'Normalization':
                            params = current_params[f'normalization_{idx}']
                            for method in params['methods']:
                                if method == "SNV":
                                    spectral_data.snv()
                                elif method == "MSC":
                                    spectral_data.msc()
                                elif method == "Detrend":
                                    p = params['parameters'][f'Detrend_{idx}']
                                    spectral_data.detrend(order=p['order'])
                                elif method == "Area":
                                    spectral_data.area()
                                elif method == "Peak Normalization":
                                    p = params['parameters'][f'Peak_{idx}']
                                    spectral_data.peaknorm(wavenumber=p['wave'])
                                elif method == "Vector":
                                    spectral_data.vector()
                                elif method == "Min-max":
                                    p = params['parameters'][f'Minmax_{idx}']
                                    spectral_data.minmax(min_val=p['minv'], max_val=p['maxv'])
                                elif method == "Pareto":
                                    spectral_data.pareto()
                                applied_steps.append(f"Normalization: {method}")

                        elif step_name == 'Center':
                            params = current_params[f'center_{idx}']
                            for method in params['methods']:
                                if method == 'Mean (spectrum)':
                                    spectral_data.mean_center(option=False)
                                elif method == 'Mean (wavelength)':
                                    spectral_data.mean_center(option=True)
                                elif method == 'Last Point':
                                    spectral_data.lastpoint()
                                applied_steps.append(f"Center: {method}")

                        elif step_name == 'Derivative':
                            params = current_params[f'derivative_{idx}']
                            for option in params['options']:
                                if option == "Subtract":
                                    spectral_data.subtract(spectra=params['parameters']['subtract_idx'])
                                elif option == "Reset":
                                    spectral_data.reset()
                                applied_steps.append(f"Derivative: {option}")

                        elif step_name == 'SG Derivative':
                            params = current_params[f'sg_derivative_{idx}']
                            spectral_data.SGDeriv(
                                window=int(params['window']),
                                poly=int(params['poly']),
                                order=int(params['order'])
                            )
                            applied_steps.append(f"SG Derivative: order={params['order']}")

                        st.success(f"Applied {step_name}")

                    st.session_state.X = spectral_data.spc.copy()
                    st.session_state.preprocessing_history.extend(applied_steps)
                    st.session_state.preprocessing_params['spectral_steps'] = applied_steps
                    st.session_state.standard_preprocessing_done = True

                    st.success("Manual preprocessing completed successfully")
                    st.info(f"Data shape: {spectral_data.spc.shape}")

                    fig, ax = plt.subplots(figsize=(12, 6))
                    for i in range(min(100, len(spectral_data.spc))):
                        ax.plot(spectral_data.wav[:spectral_data.spc.shape[1]], spectral_data.spc.iloc[i], alpha=0.6)
                    ax.set_title("Processed Spectra")
                    ax.set_xlabel("Wavelength")
                    ax.set_ylabel("Intensity")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close(fig)

                except Exception as e:
                    st.error(f"Error in manual preprocessing: {str(e)}")

        else:
            st.subheader("Automated Preprocessing Optimization")

            col1, col2 = st.columns(2)
            with col1:
                n_trials = st.selectbox("Number of Trials", [10, 25, 50, 100, 150, 200], index=1)
            with col2:
                cv_folds = st.selectbox("CV Folds", [3, 5, 7, 10], index=1)

            if st.button("START AUTOMATED PREPROCESSING", type="primary"):
                try:
                    with st.spinner(f"Running automated {technique} preprocessing..."):
                        X_current = spectral_data.spc.values
                        y_data = y.values.ravel() if len(y.shape) > 1 else y.values

                        if technique == "Raman Spectroscopy":
                            optimizer = RamanPureOptimizer(
                                X=X_current,
                                y=y_data,
                                cv_folds=cv_folds,
                                n_trials=n_trials,
                                random_state=42
                            )
                        elif technique == "NIR Spectroscopy":
                            optimizer = NIRPreprocessingOptimizer(
                                X=X_current,
                                y=y_data,
                                cv_folds=cv_folds,
                                n_trials=n_trials,
                                random_state=42
                            )
                        elif technique == "FTIR Spectroscopy":
                            optimizer = FTIRPreprocessingOptimizer(
                                X=X_current,
                                y=y_data,
                                cv_folds=cv_folds,
                                n_trials=n_trials,
                                random_state=42
                            )
                        elif technique == "Mass Spectrometry":
                            optimizer = MassSpectralPreprocessingOptimizer(
                                X=X_current,
                                y=y_data,
                                cv_folds=cv_folds,
                                n_trials=n_trials,
                                random_state=42
                            )
                        else:
                            st.error(f"Optimizer for '{technique}' is not yet implemented.")
                            return

                        results = optimizer.optimize()

                        if not results.get('success', False):
                            st.error(f"Optimization failed: {results.get('error', 'Unknown error')}")
                            return

                        X_processed = optimizer.apply_best_preprocessing(X_current, fit_mode=True)

                        processed_spectral_data = copy.deepcopy(spectral_data)
                        n_features = X_processed.shape[1]

                        if n_features <= len(spectral_data.spc.columns):
                            processed_spectral_data.spc = pd.DataFrame(X_processed, columns=spectral_data.spc.columns[:n_features])
                            processed_spectral_data.wav = spectral_data.wav[:n_features]
                        else:
                            new_columns = [f"feature_{i}" for i in range(n_features)]
                            processed_spectral_data.spc = pd.DataFrame(X_processed, columns=new_columns)
                            processed_spectral_data.wav = np.arange(n_features)

                        st.session_state.spectral_data = processed_spectral_data
                        st.session_state.X = processed_spectral_data.spc.copy()

                        technique_short = technique.split()[0]
                        automated_steps = [f"Auto-{technique_short}-{i+1}: {step['method']}"
                                        for i, step in enumerate(results['best_pipeline'])]
                        st.session_state.preprocessing_history.extend(automated_steps)
                        st.session_state.preprocessing_params['automated_technique'] = technique
                        st.session_state.preprocessing_params['automated_pipeline'] = results['best_pipeline']
                        st.session_state.preprocessing_params['automated_optimizer_info'] = {
                            'technique': technique,
                            'cv_folds': cv_folds,
                            'n_trials': n_trials,
                            'best_params': results['best_params'],
                            'cv_score': results['cv_score']
                        }
                        st.session_state.standard_preprocessing_done = True

                        st.success(f"Automated {technique} preprocessing completed")
                        st.metric("Best CV R² Score", f"{results['cv_score']:.4f}")

                        st.write("Optimal Preprocessing Pipeline:")
                        for i, step in enumerate(results['best_pipeline']):
                            st.write(f"{i+1}. {step['method']} - {step.get('params', {})}")

                        fig = plot_spectra_comparison(
                            spectral_data.spc, processed_spectral_data.spc,
                            spectral_data.wav, processed_spectral_data.wav,
                            f"Automated {technique} Preprocessing"
                        )
                        st.pyplot(fig)
                        plt.close(fig)

                except Exception as e:
                    st.error(f"Error in automated preprocessing: {str(e)}")

        if st.session_state.get('standard_preprocessing_done', False):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Proceed to Advanced Preprocessing"):
                    st.session_state.current_step = 6
                    st.rerun()
            with col2:
                if st.button("Skip to Export Results"):
                    st.session_state.current_step = 7
                    st.rerun()

    elif st.session_state.current_step == 6:
        st.header("Step 6: Advanced Preprocessing")

        if not st.session_state.target_set:
            st.error("Please complete previous steps first")
            return

        X = st.session_state.X
        y = st.session_state.y

        if 'fft_applied' not in st.session_state:
            st.session_state.fft_applied = False
        if 'opls_applied' not in st.session_state:
            st.session_state.opls_applied = False

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Samples", X.shape[0])
            st.metric("Current Features", X.shape[1])
        with col2:
            if st.session_state.fft_applied:
                st.success("FFT Applied")
            if st.session_state.opls_applied:
                st.success("OPLS Applied")

        st.subheader("FFT Filtering")

        fft_col1, fft_col2 = st.columns(2)
        with fft_col1:
            fft_threshold = st.number_input("Frequency Threshold", value=1000.0, min_value=1.0, max_value=10000.0)
        with fft_col2:
            fft_sampling_interval = st.number_input("Sampling Interval", value=0.02, min_value=0.001, max_value=1.0, format="%.4f")

        if st.button("Apply FFT Filtering"):
            try:
                with st.spinner("Applying FFT filtering..."):
                    fft_processor = FFTProcessor(
                        X, X, y, y,
                        threshold=fft_threshold,
                        sampling_interval=fft_sampling_interval
                    )

                    fft_processor.fit()
                    fft_filtered, _ = fft_processor.transform()

                    fft_df = pd.DataFrame(
                        fft_filtered,
                        columns=X.columns if fft_filtered.shape[1] == X.shape[1] else [f'feature_{i}' for i in range(fft_filtered.shape[1])],
                        index=X.index
                    )

                    st.session_state.X_original = X.copy()
                    st.session_state.X = fft_df
                    st.session_state.fft_applied = True

                    st.session_state.preprocessing_params['fft'] = {
                        'threshold': float(fft_threshold),
                        'sampling_interval': float(fft_sampling_interval),
                        'applied': True
                    }
                    st.session_state.preprocessing_history.append(f"FFT Filtering: threshold={fft_threshold:.0f}")

                    st.success("FFT filtering applied successfully")
                    st.info(f"Data shape: {X.shape} to {fft_df.shape}")

            except Exception as e:
                st.error(f"Error in FFT filtering: {str(e)}")

        st.markdown("---")
        st.subheader("OPLS Analysis")

        opls_col1, opls_col2 = st.columns(2)
        with opls_col1:
            n_opls_components = st.slider("Number of Orthogonal Components", 1, min(20, X.shape[1]), 5)
        with opls_col2:
            opls_scale_data = st.checkbox("Scale Data", True)

        if st.button("Apply OPLS Analysis"):
            try:
                with st.spinner("Applying OPLS analysis..."):
                    opls = OPLS(X.values, X.values, y, y, n_components=n_opls_components, scale=opls_scale_data)

                    opls.fit()
                    opls_filtered = opls.transform(X.values)

                    opls_df = pd.DataFrame(
                        opls_filtered,
                        columns=X.columns if opls_filtered.shape[1] == X.shape[1] else [f'feature_{i}' for i in range(opls_filtered.shape[1])],
                        index=X.index
                    )

                    if not hasattr(st.session_state, 'X_original'):
                        st.session_state.X_original = X.copy()
                    st.session_state.X = opls_df
                    st.session_state.opls_applied = True

                    st.session_state.preprocessing_params['opls'] = {
                        'n_components': int(n_opls_components),
                        'scale': bool(opls_scale_data),
                        'applied': True
                    }
                    st.session_state.preprocessing_history.append(f"OPLS: {n_opls_components} orthogonal components")

                    r2x_score = opls.score(X.values)

                    st.success(f"OPLS analysis completed")
                    st.metric("R²X Score", f"{r2x_score:.4f}")
                    st.info(f"Data shape: {X.shape} to {opls_df.shape}")

            except Exception as e:
                st.error(f"Error in OPLS analysis: {str(e)}")

        st.markdown("---")
        if st.button("Proceed to Export Results"):
            st.session_state.current_step = 7
            st.rerun()

    elif st.session_state.current_step == 7:
        st.header("Step 7: Export Results")

        if not st.session_state.target_set:
            st.error("Please complete previous steps first")
            return

        X_final = st.session_state.X
        y_final = st.session_state.y

        st.success("Preprocessing completed")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Final Data Shape", f"{X_final.shape[0]} × {X_final.shape[1]}")
        with col2:
            st.metric("Total Preprocessing Steps", len(st.session_state.preprocessing_history))

        st.subheader("Preprocessing History")
        for i, step in enumerate(st.session_state.preprocessing_history):
            st.write(f"{i+1}. {step}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        user_name_clean = re.sub(r'[^a-zA-Z0-9_]', '_', st.session_state.user_name)

        final_params = {
            "user_info": {
                "user_name": st.session_state.user_name,
                "creation_date": datetime.now().isoformat(),
                "user_id": user_name_clean
            },
            "preprocessing_parameters": st.session_state.preprocessing_params,
            "preprocessing_history": st.session_state.preprocessing_history,
            "data_info": {
                "final_shape": list(X_final.shape),
                "target_columns": st.session_state.target_columns,
                "feature_columns": list(X_final.columns)
            }
        }

        json_filename = f"preprocessing_params_{user_name_clean}_{timestamp}.json"
        save_parameters_to_json(final_params, json_filename)

        st.subheader("Download Files")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Preprocessed Data")
            X_csv_filename = f"preprocessed_X_{user_name_clean}_{timestamp}.csv"
            X_final.to_csv(X_csv_filename, index=False)
            st.markdown(create_download_link(X_final, X_csv_filename, "Download X (Features)"), unsafe_allow_html=True)

            y_csv_filename = f"preprocessed_y_{user_name_clean}_{timestamp}.csv"
            y_final.to_csv(y_csv_filename, index=False)
            st.markdown(create_download_link(y_final, y_csv_filename, "Download y (Targets)"), unsafe_allow_html=True)

        with col2:
            st.markdown("### Parameters")
            st.markdown(create_download_link(json_filename, json_filename, "Download Parameters JSON"), unsafe_allow_html=True)

        st.subheader("Generate Comparison Figure")

        if st.button("Generate and Download Figure"):
            try:
                if hasattr(st.session_state, 'X_original'):
                    original_data = st.session_state.X_original
                else:
                    original_data = st.session_state.original_data.drop(columns=st.session_state.target_columns)

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                n_plot = min(10, len(original_data))
                for i in range(n_plot):
                    ax1.plot(original_data.iloc[i].values, alpha=0.6)
                ax1.set_title("Original Data")
                ax1.set_xlabel("Feature Index")
                ax1.set_ylabel("Intensity")
                ax1.grid(True, alpha=0.3)

                for i in range(min(10, len(X_final))):
                    ax2.plot(X_final.iloc[i].values, alpha=0.6)
                ax2.set_title("Preprocessed Data")
                ax2.set_xlabel("Feature Index")
                ax2.set_ylabel("Intensity")
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()

                fig_filename = f"preprocessing_comparison_{user_name_clean}_{timestamp}.png"
                plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
                st.pyplot(fig)
                plt.close(fig)

                st.markdown(create_download_link(fig_filename, fig_filename, "Download Comparison Figure"), unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error generating figure: {str(e)}")

        st.success("All preprocessing completed. Files are ready for download.")

        if st.button("Start New Preprocessing"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()

from chatbot import render_chatbot
render_chatbot("02_Preprocessing")

