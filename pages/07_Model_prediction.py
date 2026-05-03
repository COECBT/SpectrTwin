import json
import os
import pickle
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from preprocess import SpectralData
from midel import ReadingData, WaveletDenoiser


def apply_preprocessing(data, preprocessing_params, fitted_objects=None):
    processed_data = data.copy()
    processed_data = processed_data.astype(float)

    if fitted_objects is None:
        fitted_objects = {}

    if 'wavelet' in preprocessing_params and preprocessing_params['wavelet'].get('applied', False):
        st.write("  → Applying wavelet denoising...")
        params = preprocessing_params['wavelet']

        fitted_denoiser = fitted_objects.get('wavelet_denoiser')
        if fitted_denoiser is not None:
            processed_data = fitted_denoiser.transform(processed_data)
            st.write("    Used fitted wavelet denoiser from training.")
        else:
            denoiser = WaveletDenoiser(
                wavelet=params['wavelet'],
                level=params['level'],
                threshold_mode=params['threshold_mode']
            )
            denoiser.fitted_threshold_ = params['fitted_threshold']
            processed_data = denoiser.transform(processed_data)
            st.write("    Reconstructed wavelet denoiser from parameters.")

    trim_steps = [s for s in preprocessing_params.get('spectral_steps', [])
                  if s.startswith('Trim:') or s.startswith('Inverse Trim:')]
    if trim_steps:
        st.write("  → Applying trimming...")
        trim_data = processed_data.copy()
        try:
            trim_data.columns = [float(c) for c in trim_data.columns]
        except (ValueError, TypeError):
            pass

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            temp_path = tmp.name
            trim_data.to_csv(temp_path, index=False)

        try:
            spectral = SpectralData(temp_path)
            for step in trim_steps:
                if step.startswith('Trim:'):
                    range_part = step.split(': ')[1]
                    start, end = map(float, range_part.split(' - '))
                    spectral.trim(start=start, end=end)
                    st.write(f"    Trim: {start} – {end}")
                elif step.startswith('Inverse Trim:'):
                    range_part = step.split(': ')[1]
                    start, end = map(float, range_part.split(' - '))
                    spectral.invtrim(start=start, end=end)
                    st.write(f"    Inverse Trim: {start} – {end}")
            processed_data = spectral.spc.copy()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    if 'automated_technique' in preprocessing_params and 'automated_pipeline' in preprocessing_params:
        technique = preprocessing_params['automated_technique']
        st.write(f"  → Applying automated {technique} preprocessing pipeline...")

        auto_optimizer = fitted_objects.get('auto_optimizer')
        if auto_optimizer is not None and hasattr(auto_optimizer, 'apply_best_preprocessing'):
            if hasattr(processed_data, 'values'):
                result = auto_optimizer.apply_best_preprocessing(processed_data.values, fit_mode=False)
            else:
                result = auto_optimizer.apply_best_preprocessing(np.array(processed_data), fit_mode=False)

            if result.shape[1] <= processed_data.shape[1]:
                processed_data = pd.DataFrame(result, columns=processed_data.columns[:result.shape[1]])
            else:
                processed_data = pd.DataFrame(result, columns=[f"feature_{i}" for i in range(result.shape[1])])
            st.write("    Used fitted auto_optimizer from training.")
        else:
            prep_data = processed_data.copy()
            try:
                prep_data.columns = [float(c) for c in prep_data.columns]
            except (ValueError, TypeError):
                pass

            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                temp_path = tmp.name
                prep_data.to_csv(temp_path, index=False)

            try:
                spectral = SpectralData(temp_path)
                if hasattr(spectral.spc, 'columns'):
                    try:
                        spectral.spc.columns = spectral.spc.columns.astype(float)
                        spectral.wav = spectral.spc.columns.copy()
                        spectral._wav_raw = spectral.spc.columns.copy()
                    except (ValueError, TypeError):
                        pass

                for step in preprocessing_params['automated_pipeline']:
                    method = step['method']
                    params = step.get('params', {})
                    st.write(f"    Applying {method}...")
                    _apply_spectral_method(spectral, method, params)

                processed_data = spectral.spc.copy()
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

    elif 'spectral_steps' in preprocessing_params and 'spectral_parameters' in preprocessing_params:
        spectral_params = preprocessing_params.get('spectral_parameters', {})
        if spectral_params:
            st.write("  → Applying manual spectral preprocessing...")
            manual_data = processed_data.copy()
            try:
                manual_data.columns = [float(c) for c in manual_data.columns]
            except (ValueError, TypeError):
                pass

            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                temp_path = tmp.name
                manual_data.to_csv(temp_path, index=False)

            try:
                spectral = SpectralData(temp_path)
                _apply_manual_spectral_steps(spectral, spectral_params)
                processed_data = spectral.spc.copy()
                processed_data = processed_data.astype(float)
                if hasattr(spectral, 'wav') and len(spectral.wav) == processed_data.shape[1]:
                    processed_data.columns = [float(w) for w in spectral.wav]
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

    if 'dimensionality_parameters' in preprocessing_params and 'dimensionality_steps' in preprocessing_params:
        st.write("  → Applying dimensionality reduction...")
        dim_params = preprocessing_params['dimensionality_parameters']

        fitted_scaler = fitted_objects.get('dim_reducer_scaler')
        fitted_reducer = fitted_objects.get('dim_reducer_reducer')

        technique_order = _build_technique_order(dim_params)

        for technique, idx, key in technique_order:
            params = dim_params[key]
            st.write(f"    Applying {technique} (step {idx + 1})...")

            if technique == 'Scaling':
                if fitted_scaler is not None:
                    vals = processed_data.values if hasattr(processed_data, 'values') else processed_data
                    processed_data = pd.DataFrame(
                        fitted_scaler.transform(vals),
                        columns=processed_data.columns if hasattr(processed_data, 'columns') else None
                    )
                    st.write("      Used fitted scaler from training.")
                else:
                    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
                    method = params.get('method', 'standard')
                    scaler_map = {'standard': StandardScaler, 'minmax': MinMaxScaler, 'robust': RobustScaler}
                    scaler = scaler_map.get(method, StandardScaler)()
                    vals = processed_data.values if hasattr(processed_data, 'values') else processed_data
                    processed_data = pd.DataFrame(
                        scaler.fit_transform(vals),
                        columns=processed_data.columns if hasattr(processed_data, 'columns') else None
                    )
                    st.warning(f"      Re-fitted {method} scaler on new data (no fitted scaler available).")

            elif technique == 'PCA Analysis':
                if fitted_reducer is not None and hasattr(fitted_reducer, 'transform'):
                    vals = processed_data.values if hasattr(processed_data, 'values') else processed_data
                    pca_result = fitted_reducer.transform(vals)
                    n_comp = pca_result.shape[1]
                    processed_data = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n_comp)])
                    st.write(f"      Used fitted PCA → {n_comp} components.")
                else:
                    from sklearn.decomposition import PCA
                    pca_params = params.get('parameters', {})
                    method = params.get('method', 'variance')
                    if method == 'variance':
                        pca = PCA(n_components=pca_params.get('variance_threshold', 0.95))
                    elif method == 'fixed':
                        pca = PCA(n_components=pca_params.get('n_components', 10))
                    else:
                        pca = PCA()
                    vals = processed_data.values if hasattr(processed_data, 'values') else processed_data
                    pca_result = pca.fit_transform(vals)
                    n_comp = pca_result.shape[1]
                    processed_data = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n_comp)])
                    st.warning(f"      Re-fitted PCA on new data → {n_comp} components.")

            elif technique == 'Feature Selection':
                if fitted_reducer is not None and hasattr(fitted_reducer, 'transform'):
                    vals = processed_data.values if hasattr(processed_data, 'values') else processed_data
                    selected = fitted_reducer.transform(vals)
                    processed_data = pd.DataFrame(selected)
                    st.write(f"      Used fitted feature selector → {processed_data.shape[1]} features.")
                else:
                    st.warning("      No fitted feature selector available — skipping.")

    processed_data = processed_data.astype(float)
    return processed_data


def _apply_spectral_method(spectral, method, params):
    method_map = {
        'AsLS':             lambda: spectral.AsLS(lam=params['lam'], p=params['p'], niter=params['niter']),
        'Polyfit':          lambda: spectral.polyfit(order=params['order'], niter=params['niter']),
        'Pearson':          lambda: spectral.pearson(u=params['u'], v=params['v']),
        'Rolling':          lambda: spectral.rolling(window=params['window']),
        'Savitzky-Golay':   lambda: spectral.SGSmooth(window=params['window'], poly=params['poly']),
        'SNV':              lambda: spectral.snv(),
        'MSC':              lambda: spectral.msc(),
        'Detrend':          lambda: spectral.detrend(order=params['order']),
        'Area':             lambda: spectral.area(),
        'Peak Normalization': lambda: spectral.peaknorm(wavenumber=params['wave']),
        'Vector':           lambda: spectral.vector(),
        'Min-max':          lambda: spectral.minmax(min_val=params['minv'], max_val=params['maxv']),
        'Pareto':           lambda: spectral.pareto(),
        'Mean (spectrum)':  lambda: spectral.mean_center(option=False),
        'Mean (wavelength)':lambda: spectral.mean_center(option=True),
        'Last Point':       lambda: spectral.lastpoint(),
        'Derivative_Subtract': lambda: spectral.subtract(spectra=params['subtract_idx']),
        'Derivative_Reset': lambda: spectral.reset(),
        'SG Derivative':    lambda: spectral.SGDeriv(window=params['window'], poly=params['poly'], order=params['order']),
    }
    func = method_map.get(method)
    if func:
        func()
    else:
        st.warning(f"      Unknown preprocessing method '{method}' — skipping.")


def _apply_manual_spectral_steps(spectral, spectral_params):
    technique_order = []
    for key in spectral_params.keys():
        parts = key.split('_')
        if key.startswith('trim_'):
            technique_order.append(('Trim', int(parts[1]), key))
        elif key.startswith('baseline_'):
            technique_order.append(('Baseline Correction', int(parts[1]), key))
        elif key.startswith('smoothing_'):
            technique_order.append(('Smoothing', int(parts[1]), key))
        elif key.startswith('normalization_'):
            technique_order.append(('Normalization', int(parts[1]), key))
        elif key.startswith('center_'):
            technique_order.append(('Center', int(parts[1]), key))
        elif key.startswith('derivative_'):
            technique_order.append(('Derivative', int(parts[-1]), key))
        elif key.startswith('sg_derivative_'):
            technique_order.append(('SG Derivative', int(parts[-1]), key))

    technique_order.sort(key=lambda x: x[1])

    for technique, idx, key in technique_order:
        params = spectral_params[key]
        st.write(f"    Applying {technique} (step {idx + 1})...")

        if technique == 'Trim':
            if params['type'] == "Trim":
                spectral.trim(start=params['start'], end=params['end'])
            else:
                spectral.invtrim(start=params['start'], end=params['end'])

        elif technique == 'Baseline Correction':
            for method in params['methods']:
                if method == "AsLS":
                    p = params['parameters'][f'AsLS_{idx}']
                    spectral.AsLS(lam=p['lam'], p=p['p'], niter=int(p['niter']))
                elif method == "Polyfit":
                    p = params['parameters'][f'Polyfit_{idx}']
                    spectral.polyfit(order=int(p['order']), niter=int(p['niter']))
                elif method == "Pearson":
                    p = params['parameters'][f'Pearson_{idx}']
                    spectral.pearson(u=int(p['u']), v=int(p['v']))

        elif technique == 'Smoothing':
            for method in params['methods']:
                if method == "Rolling":
                    p = params['parameters'][f'Rolling_{idx}']
                    spectral.rolling(window=int(p['window']))
                elif method == "Savitzky-Golay":
                    p = params['parameters'][f'SG_{idx}']
                    spectral.SGSmooth(window=int(p['window']), poly=int(p['poly']))

        elif technique == 'Normalization':
            for method in params['methods']:
                if method == "SNV":
                    spectral.snv()
                elif method == "MSC":
                    spectral.msc()
                elif method == "Detrend":
                    p = params['parameters'][f'Detrend_{idx}']
                    spectral.detrend(order=p['order'])
                elif method == "Area":
                    spectral.area()
                elif method == "Peak Normalization":
                    p = params['parameters'][f'Peak_{idx}']
                    spectral.peaknorm(wavenumber=p['wave'])
                elif method == "Vector":
                    spectral.vector()
                elif method == "Min-max":
                    p = params['parameters'][f'Minmax_{idx}']
                    spectral.minmax(min_val=p['minv'], max_val=p['maxv'])
                elif method == "Pareto":
                    spectral.pareto()

        elif technique == 'Center':
            for method in params['methods']:
                if method == 'Mean (spectrum)':
                    spectral.mean_center(option=False)
                elif method == 'Mean (wavelength)':
                    spectral.mean_center(option=True)
                elif method == 'Last Point':
                    spectral.lastpoint()

        elif technique == 'Derivative':
            for option in params['options']:
                if option == "Subtract":
                    spectral.subtract(spectra=params['parameters']['subtract_idx'])
                elif option == "Reset":
                    spectral.reset()

        elif technique == 'SG Derivative':
            spectral.SGDeriv(
                window=int(params['window']),
                poly=int(params['poly']),
                order=int(params['order'])
            )


def _build_technique_order(dim_params):
    order = []
    for key in dim_params.keys():
        if key.startswith('scaling_'):
            order.append(('Scaling', int(key.split('_')[-1]), key))
        elif key.startswith('pca_'):
            order.append(('PCA Analysis', int(key.split('_')[-1]), key))
        elif key.startswith('feature_selection_'):
            order.append(('Feature Selection', int(key.split('_')[-1]), key))
    order.sort(key=lambda x: x[1])
    return order


def main():

    st.title("🔮 Model Prediction")
    st.markdown("Upload your saved model files and new data to make predictions.")

    st.header("📁 Upload Files")

    col1, col2, col3 = st.columns(3)
    with col1:
        uploaded_model = st.file_uploader("Model file (.pkl)", type=["pkl"], key="pred_model")
    with col2:
        uploaded_params = st.file_uploader("Parameters file (.json)", type=["json"], key="pred_params")
    with col3:
        uploaded_fitted = st.file_uploader("Fitted objects (.pkl, optional)", type=["pkl"], key="pred_fitted")

    st.header("📊 Upload New Data")
    uploaded_data = st.file_uploader("New data for prediction", type=["csv", "xlsx", "txt"], key="pred_data")

    skip_preprocessing = st.checkbox("Skip preprocessing (data is already preprocessed)", value=False)

    if uploaded_model and uploaded_params and uploaded_data:
        if st.button("🚀 Run Prediction", type="primary", use_container_width=True):
            try:
                with st.spinner("Loading model..."):
                    model = pickle.load(uploaded_model)

                with st.spinner("Loading parameters..."):
                    parameters = json.load(uploaded_params)

                model_info = parameters.get('model_info', {})
                model_params = parameters.get('model_parameters', {})
                user_info = parameters.get('user_info', {})
                data_info = parameters.get('data_info', {})

                st.subheader("📋 Model Information")
                info_col1, info_col2, info_col3 = st.columns(3)
                with info_col1:
                    st.metric("Model Name", model_info.get('model_name', model_params.get('model_name', 'Unknown')))
                with info_col2:
                    st.metric("Model Type", model_params.get('model_type', 'Unknown'))
                with info_col3:
                    st.metric("Created by", user_info.get('user_name', 'Unknown'))

                if 'performance_metrics' in model_params:
                    st.write("**Training Performance:**")
                    metric_cols = st.columns(min(4, len(model_params['performance_metrics'])))
                    for i, (metric, value) in enumerate(model_params['performance_metrics'].items()):
                        if isinstance(value, (int, float)):
                            with metric_cols[i % len(metric_cols)]:
                                st.metric(metric, f"{value:.4f}")

                fitted_objects = {}
                if uploaded_fitted:
                    with st.spinner("Loading fitted objects..."):
                        fitted_objects = pickle.load(uploaded_fitted)
                    st.success(f"Loaded fitted objects: {list(fitted_objects.keys())}")

                with st.spinner("Loading data..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_data.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_data.getbuffer())
                        file_path = tmp_file.name

                    rd = ReadingData()
                    new_data = rd.read_data(file_path)
                    os.unlink(file_path)

                st.write(f"**Data shape:** {new_data.shape}")

                if skip_preprocessing:
                    st.info("Skipping preprocessing")
                    processed_data = new_data.astype(float)
                else:
                    preprocessing_params = parameters.get('preprocessing_parameters', {})
                    if preprocessing_params:
                        with st.spinner("Applying preprocessing pipeline..."):
                            processed_data = apply_preprocessing(new_data, preprocessing_params, fitted_objects)
                        st.success(f"Preprocessed data shape: {processed_data.shape}")
                    else:
                        st.warning("No preprocessing parameters found — using raw data.")
                        processed_data = new_data.astype(float)

                expected_features = len(data_info.get('feature_columns', []))
                if expected_features > 0:
                    if processed_data.shape[1] != expected_features:
                        st.warning(f"Feature mismatch! Expected {expected_features}, got {processed_data.shape[1]}")
                    else:
                        st.success(f"Feature count matches: {expected_features}")

                with st.spinner("Making predictions..."):
                    if hasattr(processed_data, 'values'):
                        prediction_data = processed_data.values.astype(float)
                    else:
                        prediction_data = np.array(processed_data, dtype=float)

                    if np.any(np.isnan(prediction_data)) or np.any(np.isinf(prediction_data)):
                        st.error("Data contains NaN or Inf values after preprocessing!")
                        return

                    is_two_part = (isinstance(model, dict) and
                                   'classification_model' in model and
                                   'regression_model' in model)

                    if is_two_part:
                        clf_model = model['classification_model']
                        reg_model = model['regression_model']

                        binary_predictions = clf_model.predict(prediction_data)
                        predictions = np.zeros(len(prediction_data), dtype=float)

                        nonzero_mask = binary_predictions == 1
                        if np.any(nonzero_mask):
                            reg_predictions = reg_model.predict(prediction_data[nonzero_mask])
                            predictions[nonzero_mask] = reg_predictions

                        st.info(f"Two-part model: {int(np.sum(binary_predictions == 0))} zero, {int(np.sum(binary_predictions == 1))} non-zero")
                    else:
                        predictions = model.predict(prediction_data)

                    model_type = model_params.get('model_type', '')
                    if model_type in ('regression', 'zero_inflated'):
                        neg_count = int(np.sum(predictions < 0))
                        if neg_count > 0:
                            st.warning(f"Clipped {neg_count} negative predictions to 0.")
                            predictions = np.maximum(predictions, 0)

                target_columns = data_info.get('target_columns', [])

                if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                    results_df = pd.DataFrame(predictions)
                    results_df.insert(0, 'Sample_Index', range(len(predictions)))
                    if target_columns and len(target_columns) == predictions.shape[1]:
                        results_df.columns = ['Sample_Index'] + target_columns
                    else:
                        results_df.columns = ['Sample_Index'] + [f'Target_{i+1}' for i in range(predictions.shape[1])]
                else:
                    predictions_flat = predictions.flatten() if len(predictions.shape) > 1 else predictions
                    target_name = target_columns[0] if target_columns and len(target_columns) == 1 else 'Prediction'
                    results_df = pd.DataFrame({
                        'Sample_Index': range(len(predictions_flat)),
                        target_name: predictions_flat
                    })

                st.header("📊 Prediction Results")

                pred_values = predictions.flatten() if len(predictions.shape) > 1 else predictions

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Predictions", len(pred_values))
                with col2:
                    st.metric("Mean", f"{pred_values.mean():.4f}")
                with col3:
                    st.metric("Std Dev", f"{pred_values.std():.4f}")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Min", f"{pred_values.min():.4f}")
                with col2:
                    st.metric("Max", f"{pred_values.max():.4f}")
                with col3:
                    st.metric("Median", f"{np.median(pred_values):.4f}")

                st.dataframe(results_df, use_container_width=True)

                import matplotlib.pyplot as plt

                if results_df.shape[1] == 2:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                    ax1.plot(results_df['Sample_Index'], results_df.iloc[:, 1], 'o-', alpha=0.7, linewidth=2, markersize=4)
                    ax1.set_xlabel('Sample Index')
                    ax1.set_ylabel('Predicted Value')
                    ax1.set_title('Predictions on New Data')
                    ax1.grid(True, alpha=0.3)

                    ax2.hist(results_df.iloc[:, 1], bins=min(30, max(5, len(results_df)//5)), alpha=0.7, edgecolor='black')
                    ax2.set_xlabel('Predicted Value')
                    ax2.set_ylabel('Frequency')
                    ax2.set_title('Distribution of Predictions')
                    ax2.grid(True, alpha=0.3)

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Predictions as CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    else:
        st.info("Please upload the **Model (.pkl)**, **Parameters (.json)**, and **New Data** files to begin.")


main()


from chatbot import render_chatbot
render_chatbot("07_Model_prediction")

