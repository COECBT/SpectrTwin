import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tempfile
import os
from scipy import signal
from scipy.stats import zscore



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



def app():


    # Initialize session state
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
    if 'skipped_steps' not in st.session_state:
        st.session_state.skipped_steps = set()

    # ==================== STEP 1: DATA LOADING ====================
    if st.session_state.step == 1:
        st.header("Step 1: Load Spectral Data")
        
        st.subheader("User Information")
        st.success(f"Welcome, {st.session_state.user_name}!")
        
        if st.button("Change Name"):
            st.session_state.user_name = ""
            st.rerun()
        
        st.subheader("Data Upload")
        st.info("Upload your spectroscopic data file (CSV, XLSX, or TXT)")
        
        uploaded_file = st.file_uploader("Upload spectral data file", type=["csv", "xlsx", "txt"])
        
        if uploaded_file:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    file_path = tmp_file.name

                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(file_path)
                elif uploaded_file.name.endswith('.xlsx'):
                    data = pd.read_excel(file_path)
                elif uploaded_file.name.endswith('.txt'):
                    data = pd.read_csv(file_path, delimiter='\t')
                
                st.session_state.original_data = data.copy()
                st.session_state.current_data = data.copy()
                st.session_state.data_loaded = True
                
                st.success("Data loaded successfully!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Samples", data.shape[0])
                    st.metric("Features", data.shape[1])
                with col2:
                    st.metric("Missing Values", data.isnull().sum().sum())
                    st.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                
                st.dataframe(data.head(10))
                
                os.unlink(file_path)
                
                if st.button("Proceed to Target Selection →"):
                    st.session_state.step = 2
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")

    # ==================== STEP 2: TARGET SELECTION ====================
    elif st.session_state.step == 2:
        st.header("Step 2: Target Selection")

        if not st.session_state.data_loaded:
            st.error("Please upload data first")
            if st.button("← Go back to Data Upload"):
                st.session_state.step = 1
                st.rerun()
            st.stop()

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
                for key in ['X_full', 'y_full', 'current_X', 'target_columns', 'x_axis']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("Dataset reset to original")
                st.rerun()

        st.dataframe(data.head(10))

        st.subheader("Target Column Selection")
        target_columns = st.multiselect("Select Target Columns (e.g., concentration, class labels)", data.columns)
        
        if st.button("Set Targets and Continue"):
            if target_columns:
                st.session_state.target_columns = target_columns
                st.session_state.targets_set = True
                X = data.drop(columns=target_columns)
                y = data[target_columns]

                st.session_state.X_full = X
                st.session_state.y_full = y
                st.session_state.current_X = X.copy()

                # Try to convert column names to float (wavelengths/wavenumbers)
                try:
                    x_axis = X.columns.astype(float)
                    st.session_state.x_axis = x_axis
                    st.session_state.x_axis_label = "Wavelength / Wavenumber"
                except (ValueError, TypeError):
                    st.session_state.x_axis = np.arange(X.shape[1])
                    st.session_state.x_axis_label = "Feature Index"
                    st.warning("Column names cannot be converted to numeric. Using indices for plotting.")

                st.success(f"Target columns set: {', '.join(target_columns)}")
                st.info(f"Spectral Features: {X.shape[1]} | Samples: {X.shape[0]} | Targets: {y.shape[1]}")
                
                st.session_state.step = 3
                st.rerun()
            else:
                st.error("Please select at least one target column")

        if st.button("← Back to Data Upload"):
            st.session_state.step = 1
            st.rerun()

    # ==================== STEP 3: DATA VISUALIZATION ====================
    elif st.session_state.step == 3:
        st.header("Step 3: Spectral Data Visualization")
        
        if not st.session_state.targets_set:
            st.error("Please set targets first")
            if st.button("← Go back to Target Selection"):
                st.session_state.step = 2
                st.rerun()
            st.stop()

        X = st.session_state.X_full
        y = st.session_state.y_full
        x_axis = st.session_state.x_axis
        x_label = st.session_state.x_axis_label

        # Sidebar controls
        st.sidebar.header("Visualization Controls")
        
        # Wavelength/wavenumber range selection
        st.sidebar.subheader("Spectral Range")
        min_val, max_val = float(x_axis.min()), float(x_axis.max())
        
        if min_val == max_val:
            st.sidebar.warning("All wavelength/wavenumber values are identical")
            range_selection = (min_val, max_val)
        else:
            range_selection = st.sidebar.slider(
                "Select range",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val)
            )
        
        # Filter data based on range
        mask = (x_axis >= range_selection[0]) & (x_axis <= range_selection[1])
        x_axis_filtered = x_axis[mask]
        X_filtered = X.iloc[:, mask]
        
        # Sample selection
        st.sidebar.subheader("Sample Selection")
        n_samples = X.shape[0]
        sample_selection_method = st.sidebar.radio("Select samples by:", ["All", "Range", "Random", "Specific Indices"])
        
        if sample_selection_method == "Range":
            if n_samples == 1:
                selected_samples = [0]
                st.sidebar.info("Only 1 sample available")
            else:
                sample_range = st.sidebar.slider("Sample range", 0, n_samples-1, (0, min(19, n_samples-1)))
                selected_samples = list(range(sample_range[0], sample_range[1]+1))
        elif sample_selection_method == "Random":
            n_random = st.sidebar.number_input("Number of random samples", 1, n_samples, min(10, n_samples))
            if st.sidebar.button("Generate Random Samples"):
                st.session_state.random_samples = np.random.choice(n_samples, n_random, replace=False)
            selected_samples = st.session_state.get('random_samples', list(range(min(10, n_samples))))
        elif sample_selection_method == "Specific Indices":
            indices_input = st.sidebar.text_input("Enter indices (comma-separated)", "0,1,2,3,4")
            try:
                selected_samples = [int(i.strip()) for i in indices_input.split(',') if i.strip()]
                selected_samples = [i for i in selected_samples if 0 <= i < n_samples]
            except:
                selected_samples = list(range(min(5, n_samples)))
                st.sidebar.error("Invalid indices, using default")
        else:  # All
            selected_samples = list(range(n_samples))

        # Display basic statistics
        st.subheader(" Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Samples", X.shape[0])
        with col2:
            st.metric("Spectral Points", X.shape[1])
        with col3:
            st.metric("Target Variables", y.shape[1])
        with col4:
            st.metric("Selected Range", f"{x_axis_filtered.shape[0]} points")

        # Statistics summary
        with st.expander("Statistical Summary"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Spectral Data (X)**")
                st.write(f"Missing Values: {X.isnull().sum().sum()}")
                st.write(f"Intensity Range: [{X.min().min():.4f}, {X.max().max():.4f}]")
                st.write(f"Mean Intensity: {X.mean().mean():.4f}")
                st.write(f"Std Intensity: {X.std().mean():.4f}")
            with col2:
                st.write("**Target Data (y)**")
                st.write(f"Missing Values: {y.isnull().sum().sum()}")
                st.write(y.describe())

        # Visualization tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            " Individual Spectra", 
            " Overlay Plot", 
            " Heatmap", 
            " Statistics",
            " Target Analysis",
            " Peak Detection"
        ])

        # TAB 1: Individual Spectra
        with tab1:
            st.subheader("Individual Spectrum Viewer")
            sample_idx = st.selectbox("Select sample to view", selected_samples)
            
            fig = go.Figure()
            spectrum = X_filtered.iloc[sample_idx]
            
            fig.add_trace(go.Scatter(
                x=x_axis_filtered,
                y=spectrum,
                mode='lines',
                name=f'Sample {sample_idx}',
                line=dict(width=2)
            ))
            
            target_info = " | ".join([f"{col}: {y.iloc[sample_idx][col]}" for col in y.columns])
            
            fig.update_layout(
                title=f"Spectrum - Sample {sample_idx}<br><sub>{target_info}</sub>",
                xaxis_title=x_label,
                yaxis_title="Intensity",
                hovermode='x unified',
                height=500,
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show raw data
            with st.expander("View Raw Spectrum Data"):
                spectrum_df = pd.DataFrame({
                    x_label: x_axis_filtered,
                    'Intensity': spectrum.values
                })
                st.dataframe(spectrum_df)

        # TAB 2: Overlay Plot
        with tab2:
            st.subheader("Multi-Spectrum Overlay")
            
            # Handle edge case where there are few samples
            n_selected = len(selected_samples)
            if n_selected <= 5:
                max_display = n_selected
                st.info(f"Displaying all {n_selected} selected samples")
            else:
                max_display_max = max(50, n_selected)
                max_display_default = max(10, n_selected)
                max_display = st.slider(
                    "Maximum spectra to display", 
                    1, 
                    max_display_max, 
                    max_display_default
                )
            
            display_samples = selected_samples[:max_display]
            
            color_by = st.selectbox("Color spectra by:", ["Index"] + list(y.columns))
            
            fig = go.Figure()
            
            if color_by == "Index":
                colors = px.colors.sample_colorscale("viridis", [i/(len(display_samples)-1) for i in range(len(display_samples))])
                for i, idx in enumerate(display_samples):
                    fig.add_trace(go.Scatter(
                        x=x_axis_filtered,
                        y=X_filtered.iloc[idx],
                        mode='lines',
                        name=f'Sample {idx}',
                        line=dict(width=1.5, color=colors[i]),
                        opacity=0.7
                    ))
            else:
                # Color by target variable
                target_values = y[color_by].iloc[display_samples]
                for idx in display_samples:
                    fig.add_trace(go.Scatter(
                        x=x_axis_filtered,
                        y=X_filtered.iloc[idx],
                        mode='lines',
                        name=f'Sample {idx} ({y[color_by].iloc[idx]})',
                        line=dict(width=1.5),
                        opacity=0.7
                    ))
            
            fig.update_layout(
                title=f"Overlay of {len(display_samples)} Spectra",
                xaxis_title=x_label,
                yaxis_title="Intensity",
                hovermode='x unified',
                height=600,
                template='plotly_white',
                showlegend=(len(display_samples) <= 10)
            )
            st.plotly_chart(fig, use_container_width=True)

        # TAB 3: Heatmap
        with tab3:
            st.subheader("Spectral Heatmap")
            
            # Handle edge case where there are few samples
            n_selected = len(selected_samples)
            if n_selected <= 5:
                max_heatmap = n_selected
                st.info(f"Displaying all {n_selected} selected samples")
            else:
                max_heatmap_max = max(10, n_selected)
                max_heatmap_default = max(20, n_selected)
                max_heatmap = st.slider(
                    "Maximum samples for heatmap", 
                    5, 
                    max_heatmap_max, 
                    max_heatmap_default
                )
            
            heatmap_samples = selected_samples[:max_heatmap]
            
            # Option to normalize
            normalize = st.checkbox("Normalize each spectrum", value=False)
            
            heatmap_data = X_filtered.iloc[heatmap_samples].copy()
            if normalize:
                heatmap_data = heatmap_data.div(heatmap_data.max(axis=1), axis=0)
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=x_axis_filtered,
                y=[f"Sample {i}" for i in heatmap_samples],
                colorscale='Viridis',
                colorbar=dict(title="Intensity")
            ))
            
            fig.update_layout(
                title=f"Spectral Heatmap ({len(heatmap_samples)} samples)",
                xaxis_title=x_label,
                yaxis_title="Sample",
                height=max(400, len(heatmap_samples) * 15),
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)

        # TAB 4: Statistics
        with tab4:
            st.subheader("Statistical Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Mean Spectrum with Confidence Interval**")
                mean_spectrum = X_filtered.iloc[selected_samples].mean(axis=0)
                std_spectrum = X_filtered.iloc[selected_samples].std(axis=0)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x_axis_filtered,
                    y=mean_spectrum,
                    mode='lines',
                    name='Mean',
                    line=dict(color='blue', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=np.concatenate([x_axis_filtered, x_axis_filtered[::-1]]),
                    y=np.concatenate([mean_spectrum + std_spectrum, (mean_spectrum - std_spectrum)[::-1]]),
                    fill='toself',
                    fillcolor='rgba(0,100,255,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='±1 Std'
                ))
                fig.update_layout(
                    xaxis_title=x_label,
                    yaxis_title="Intensity",
                    height=400,
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**Standard Deviation Spectrum**")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x_axis_filtered,
                    y=std_spectrum,
                    mode='lines',
                    line=dict(color='red', width=2)
                ))
                fig.update_layout(
                    xaxis_title=x_label,
                    yaxis_title="Std Deviation",
                    height=400,
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Percentile analysis
            st.write("**Percentile Analysis**")
            percentiles = [10, 25, 50, 75, 90]
            fig = go.Figure()
            
            for p in percentiles:
                percentile_spectrum = X_filtered.iloc[selected_samples].quantile(p/100, axis=0)
                fig.add_trace(go.Scatter(
                    x=x_axis_filtered,
                    y=percentile_spectrum,
                    mode='lines',
                    name=f'{p}th percentile',
                    line=dict(width=1.5)
                ))
            
            fig.update_layout(
                title="Percentile Spectra",
                xaxis_title=x_label,
                yaxis_title="Intensity",
                height=500,
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)

        # TAB 5: Target Analysis
        with tab5:
            st.subheader("Target Variable Analysis")
            
            target_col = st.selectbox("Select target variable", y.columns)
            
            # Check if target is numeric or categorical
            if pd.api.types.is_numeric_dtype(y[target_col]):
                # Numeric target
                st.write("**Spectra grouped by target value ranges**")
                
                # Check if there are enough unique values for binning
                n_unique = y[target_col].nunique()
                if n_unique < 2:
                    st.warning(f"Target variable has only {n_unique} unique value(s). Cannot create meaningful bins.")
                    st.dataframe(y[target_col].value_counts())
                else:
                    max_bins = min(10, n_unique)
                    n_bins = st.slider("Number of bins", 2, max_bins, min(3, max_bins))
                    y_binned = pd.cut(y[target_col], bins=n_bins)
                    
                    fig = go.Figure()
                    for bin_label in y_binned.cat.categories:
                        bin_mask = y_binned == bin_label
                        if bin_mask.sum() > 0:
                            mean_spectrum = X_filtered[bin_mask].mean(axis=0)
                            fig.add_trace(go.Scatter(
                                x=x_axis_filtered,
                                y=mean_spectrum,
                                mode='lines',
                                name=f'{target_col}: {bin_label}',
                                line=dict(width=2)
                            ))
                    
                    fig.update_layout(
                        title=f"Mean Spectra by {target_col} Ranges",
                        xaxis_title=x_label,
                        yaxis_title="Mean Intensity",
                        height=500,
                        template='plotly_white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Scatter plot: Target vs specific wavelength
                st.write("**Target vs Intensity at specific wavelength**")
                if len(x_axis_filtered) == 1:
                    wavelength_idx = 0
                    st.info(f"Using wavelength: {x_axis_filtered[wavelength_idx]:.2f}")
                else:
                    wavelength_idx = st.slider(
                        "Select wavelength index", 
                        0, 
                        len(x_axis_filtered)-1, 
                        len(x_axis_filtered)//2
                    )
                
                fig = px.scatter(
                    x=X_filtered.iloc[:, wavelength_idx],
                    y=y[target_col],
                    labels={'x': f'Intensity at {x_axis_filtered[wavelength_idx]:.2f}', 'y': target_col},
                    trendline="ols"
                )
                fig.update_layout(height=400, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                # Categorical target
                st.write("**Mean spectra by category**")
                unique_classes = y[target_col].unique()
                
                fig = go.Figure()
                for cls in unique_classes:
                    cls_mask = y[target_col] == cls
                    mean_spectrum = X_filtered[cls_mask].mean(axis=0)
                    std_spectrum = X_filtered[cls_mask].std(axis=0)
                    
                    fig.add_trace(go.Scatter(
                        x=x_axis_filtered,
                        y=mean_spectrum,
                        mode='lines',
                        name=f'{cls} (n={cls_mask.sum()})',
                        line=dict(width=2)
                    ))
                
                fig.update_layout(
                    title=f"Mean Spectra by {target_col}",
                    xaxis_title=x_label,
                    yaxis_title="Mean Intensity",
                    height=500,
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)

        # TAB 6: Peak Detection
        with tab6:
            st.subheader("Peak Detection Analysis")
            
            sample_peak = st.selectbox("Select sample for peak detection", selected_samples, key='peak_sample')
            
            col1, col2 = st.columns(2)
            with col1:
                prominence = st.slider("Peak prominence", 0.01, 1.0, 0.1, 0.01)
            with col2:
                distance = st.slider("Minimum peak distance", 1, 50, 5)
            
            spectrum = X_filtered.iloc[sample_peak].values
            peaks, properties = signal.find_peaks(spectrum, prominence=prominence, distance=distance)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_axis_filtered,
                y=spectrum,
                mode='lines',
                name='Spectrum',
                line=dict(color='blue', width=2)
            ))
            
            if len(peaks) > 0:
                fig.add_trace(go.Scatter(
                    x=x_axis_filtered[peaks],
                    y=spectrum[peaks],
                    mode='markers',
                    name='Peaks',
                    marker=dict(color='red', size=10, symbol='x')
                ))
            
            fig.update_layout(
                title=f"Peak Detection - Sample {sample_peak} ({len(peaks)} peaks found)",
                xaxis_title=x_label,
                yaxis_title="Intensity",
                height=500,
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            if len(peaks) > 0:
                st.write(f"**Detected {len(peaks)} peaks**")
                peaks_df = pd.DataFrame({
                    'Peak Index': peaks,
                    x_label: x_axis_filtered[peaks].values,
                    'Intensity': spectrum[peaks],
                    'Prominence': properties['prominences']
                })
                st.dataframe(peaks_df)

        # Navigation buttons
        st.divider()
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("← Back to Target Selection"):
                st.session_state.step = 2
                st.rerun()



app()

from chatbot import render_chatbot
render_chatbot("01_Data_Visualization")

