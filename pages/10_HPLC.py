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
import io
import base64
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from hplc import HPLCDataProcessor, plot_chromatogram , OptimizationCalculator



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


def main():

    
    st.title("HPLC Data Processing & Optimization Pipeline")
    st.markdown("---")
    
    if 'hplc_processor' not in st.session_state:
        st.session_state.hplc_processor = HPLCDataProcessor()
    if 'opt_calculator' not in st.session_state:
        st.session_state.opt_calculator = OptimizationCalculator()
    if 'user_name' not in st.session_state:
        st.session_state.user_name = ""
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'processing_done' not in st.session_state:
        st.session_state.processing_done = False
    if 'optimization_done' not in st.session_state:
        st.session_state.optimization_done = False
    
    st.sidebar.title("Pipeline Status")
    
    steps = {
        1: "User Information",
        2: "Data Upload",
        3: "Peak Integration",
        4: "Optimization",
        5: "Timing Parameters",
        6: "Export Results"
    }
    
    for step_num, step_name in steps.items():
        if st.session_state.current_step > step_num:
            st.sidebar.success(f"Step {step_num}: {step_name}")
        elif st.session_state.current_step == step_num:
            st.sidebar.info(f"Step {step_num}: {step_name}")
        else:
            st.sidebar.write(f"Step {step_num}: {step_name}")
    
    st.sidebar.markdown("---")
    
    if 'processing_history' in st.session_state and st.session_state.processing_history:
        with st.sidebar.expander("Processing History"):
            for item in st.session_state.processing_history:
                st.write(f"- {item}")
    
    if st.session_state.current_step == 1:
        st.header("Step 1: User Information")
        
        st.info("Please enter your information to begin the HPLC data processing pipeline")
        
        user_name = st.text_input(
            "Enter Your Name:", 
            placeholder="e.g., sanjay",
            help="This will be used for file naming and tracking"
        )
        
        if user_name:
            st.session_state.user_name = user_name
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.success(f"Welcome, {user_name}!")
            with col2:
                if st.button("Proceed to Data Upload", type="primary"):
                    st.session_state.current_step = 2
                    st.session_state.processing_history = []
                    st.session_state.processing_history.append(f"User: {user_name}")
                    st.rerun()
        else:
            st.warning("Please enter your name to proceed")
    
    elif st.session_state.current_step == 2:
        st.header("Step 2: HPLC Data Upload")
        
        st.success(f"User: {st.session_state.user_name}")
        
        st.markdown("""
        ### Upload your HPLC chromatography data file
        
        **Supported formats:**
        - CSV files with various separators
        - Excel files (XLSX, XLS)
        - Text files (TXT with tab or comma delimiters)
        - Files with or without headers
        - Minimum requirement: 2 columns (retention time and signal)
        
        **Expected data structure:**
        - Time/retention column (typically column 4 or 5, or second-to-last)
        - Signal/intensity column (typically column 5 or 6, or last)
        """)
        
        uploaded_file = st.file_uploader(
            "Upload HPLC Data File", 
            type=["csv", "CSV", "xlsx", "xls", "txt", "TXT"],
            help="Upload your HPLC chromatography data"
        )
        
        if uploaded_file:
            try:
                file_extension = uploaded_file.name.split('.')[-1].lower()
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    temp_path = tmp.name
                
                st.session_state.hplc_file_path = temp_path
                st.session_state.hplc_file_name = uploaded_file.name
                
                processor = st.session_state.hplc_processor
                preview_df = processor.load_and_preview_data(temp_path, max_rows=100)
                
                st.success(f"File uploaded: {uploaded_file.name}")
                st.info(f"Data shape: {preview_df.shape[0]} rows × {preview_df.shape[1]} columns")
                
                st.subheader("Data Preview")
                st.dataframe(preview_df.head(20), use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Rows", preview_df.shape[0])
                with col2:
                    st.metric("Total Columns", preview_df.shape[1])
                with col3:
                    st.metric("File Size", f"{os.path.getsize(temp_path) / 1024:.1f} KB")
                
                st.subheader("Select Columns for Chromatogram")
                
                col1, col2 = st.columns(2)
                with col1:
                    time_col = st.selectbox(
                        "Retention Time Column",
                        options=range(len(preview_df.columns)),
                        index=min(4, len(preview_df.columns)-2),
                        format_func=lambda x: f"Column {x}: {preview_df.columns[x]}"
                    )
                with col2:
                    signal_col = st.selectbox(
                        "Signal Intensity Column",
                        options=range(len(preview_df.columns)),
                        index=min(5, len(preview_df.columns)-1),
                        format_func=lambda x: f"Column {x}: {preview_df.columns[x]}"
                    )
                
                st.session_state.time_col_idx = time_col
                st.session_state.signal_col_idx = signal_col
                
                if st.button("Preview Chromatogram"):
                    try:
                        fig = plot_chromatogram(
                            preview_df, 
                            time_col, 
                            signal_col,
                            title=f"Chromatogram Preview - {uploaded_file.name}"
                        )
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as e:
                        st.error(f"Error plotting chromatogram: {str(e)}")
                
                st.session_state.data_loaded = True
                st.session_state.processing_history.append(f"Uploaded: {uploaded_file.name}")
                
                st.markdown("---")
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("Back to User Info"):
                        st.session_state.current_step = 1
                        st.rerun()
                with col2:
                    if st.button("Proceed to Peak Integration", type="primary"):
                        st.session_state.current_step = 3
                        st.rerun()
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                with st.expander("Error Details"):
                    import traceback
                    st.code(traceback.format_exc())
        
        else:
            st.markdown("---")
            if st.button("Back to User Info"):
                st.session_state.current_step = 1
                st.rerun()
    
    elif st.session_state.current_step == 3:
        st.header("Step 3: Peak Integration Analysis")
        
        if not st.session_state.data_loaded:
            st.error("Please upload data first")
            if st.button("Go to Data Upload"):
                st.session_state.current_step = 2
                st.rerun()
            return
        
        st.success(f"Processing: {st.session_state.hplc_file_name}")
        
        processor = st.session_state.hplc_processor
        
        st.subheader("Peak Range Configuration")
        
        use_custom_ranges = st.checkbox(
            "Use custom peak ranges", 
            value=False,
            help="Override default peak ranges with custom values"
        )
        
        if use_custom_ranges:
            st.info("Enter custom retention time ranges for each peak")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Acid Peak**")
                acid_start = st.number_input("Start RT", value=8.0, key="acid_start")
                acid_end = st.number_input("End RT", value=8.738, key="acid_end")
                
                st.markdown("**Main Peak**")
                main_start = st.number_input("Start RT", value=8.738, key="main_start")
                main_end = st.number_input("End RT", value=9.153, key="main_end")
            
            with col2:
                st.markdown("**Base1 Peak**")
                base1_start = st.number_input("Start RT", value=9.153, key="base1_start")
                base1_end = st.number_input("End RT", value=9.509, key="base1_end")
                
                st.markdown("**Base2 Peak**")
                base2_start = st.number_input("Start RT", value=9.850, key="base2_start")
                base2_end = st.number_input("End RT", value=10.315, key="base2_end")
            
            custom_ranges = {
                'acid': (acid_start, acid_end),
                'main': (main_start, main_end),
                'base1': (base1_start, base1_end),
                'base2': (base2_start, base2_end),
                'total': (acid_start, base2_end)
            }
        else:
            custom_ranges = None
            st.info("Using default peak ranges")
            
            with st.expander("View Default Ranges"):
                default_ranges = processor.peak_ranges
                for peak, (start, end) in default_ranges.items():
                    if peak != 'total':
                        st.write(f"**{peak.capitalize()}**: {start} - {end} min")
        
        st.markdown("---")
        
        if st.button("Process HPLC Data", type="primary", use_container_width=True):
            with st.spinner("Processing HPLC data..."):
                results = processor.process_hplc_file(
                    st.session_state.hplc_file_path,
                    custom_ranges=custom_ranges
                )
            
            if results.get('success'):
                st.session_state.hplc_results = results
                st.session_state.processing_done = True
                st.session_state.processing_history.append("Peak integration completed")
                
                st.success("HPLC data processed successfully!")
                
                st.subheader("Peak Integration Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Acid Peak",
                        f"{results['peak_areas']['acid']:.2f}",
                        delta=f"{results['peak_percentages']['acid_pct']:.2f}%"
                    )
                
                with col2:
                    st.metric(
                        "Main Peak",
                        f"{results['peak_areas']['main']:.2f}",
                        delta=f"{results['peak_percentages']['main_pct']:.2f}%"
                    )
                
                with col3:
                    st.metric(
                        "Base1 Peak",
                        f"{results['peak_areas']['base1']:.2f}",
                        delta=f"{results['peak_percentages']['base1_pct']:.2f}%"
                    )
                
                with col4:
                    st.metric(
                        "Base2 Peak",
                        f"{results['peak_areas']['base2']:.2f}",
                        delta=f"{results['peak_percentages']['base2_pct']:.2f}%"
                    )
                
                st.metric("Total Area", f"{results['peak_areas']['total']:.2f}")
                
                with st.expander("Detailed Results Table"):
                    results_df = pd.DataFrame({
                        'Peak': ['Acid', 'Main', 'Base1', 'Base2', 'Total'],
                        'Area': [
                            results['peak_areas']['acid'],
                            results['peak_areas']['main'],
                            results['peak_areas']['base1'],
                            results['peak_areas']['base2'],
                            results['peak_areas']['total']
                        ],
                        'Percentage (%)': [
                            results['peak_percentages']['acid_pct'],
                            results['peak_percentages']['main_pct'],
                            results['peak_percentages']['base1_pct'],
                            results['peak_percentages']['base2_pct'],
                            100.0
                        ],
                        'Fraction': [
                            results['peak_fractions']['per_a'],
                            results['peak_fractions']['per_m'],
                            results['peak_fractions']['per_b1'],
                            results['peak_fractions']['per_b2'],
                            1.0
                        ]
                    })
                    st.dataframe(results_df, use_container_width=True)
                
                st.subheader("Chromatogram with Peak Regions")
                
                try:
                    preview_df = processor.load_and_preview_data(st.session_state.hplc_file_path)
                    fig = plot_chromatogram(
                        preview_df,
                        st.session_state.time_col_idx,
                        st.session_state.signal_col_idx,
                        peak_ranges=results['ranges_used'],
                        title=f"HPLC Chromatogram - {st.session_state.hplc_file_name}"
                    )
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as plot_error:
                    st.warning(f"Could not plot chromatogram: {plot_error}")
                
            else:
                st.error(f"Processing failed: {results.get('error', 'Unknown error')}")
                st.error(f"Error type: {results.get('error_type', 'Unknown')}")
        
        st.markdown("---")
        
        if st.session_state.processing_done:
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                if st.button("Back to Data Upload"):
                    st.session_state.current_step = 2
                    st.rerun()
            
            with col2:
                if st.button("Run Optimization", type="primary"):
                    st.session_state.current_step = 4
                    st.rerun()
            
            with col3:
                if st.button("Skip to Export"):
                    st.session_state.current_step = 6
                    st.rerun()
        else:
            if st.button("Back to Data Upload"):
                st.session_state.current_step = 2
                st.rerun()
    
    elif st.session_state.current_step == 4:
        st.header("Step 4: Optimization Model")
        
        if not st.session_state.processing_done:
            st.error("Please complete peak integration first")
            if st.button("Go to Peak Integration"):
                st.session_state.current_step = 3
                st.rerun()
            return
        
        st.info("This step runs an optimization model to determine optimal processing times")
        
        results = st.session_state.hplc_results
        
        st.subheader("Peak Fractions for Optimization")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Acid Fraction", f"{results['peak_fractions']['per_a']:.4f}")
        with col2:
            st.metric("Main Fraction", f"{results['peak_fractions']['per_m']:.4f}")
        with col3:
            st.metric("Base1 Fraction", f"{results['peak_fractions']['per_b1']:.4f}")
        with col4:
            st.metric("Base2 Fraction", f"{results['peak_fractions']['per_b2']:.4f}")
        
        st.markdown("---")
        
        try:
            import model_col1
            module_available = True
            st.success("Optimization module (model_col1) is available")
        except ImportError:
            module_available = False
            st.warning("Optimization module (model_col1) not found")
            st.info("You can proceed to timing parameters with manual input or skip to export")
        
        if module_available:
            if st.button("Run Optimization Model", type="primary", use_container_width=True):
                with st.spinner("Running optimization model..."):
                    calc = st.session_state.opt_calculator
                    
                    output_dir = 'tmp/simulation_files'
                    os.makedirs(output_dir, exist_ok=True)
                    
                    opt_results = calc.run_optimization_model(
                        results['peak_fractions'],
                        output_dir=output_dir,
                        csv_filename=st.session_state.hplc_file_name
                    )
                
                if opt_results.get('success'):
                    st.session_state.optimization_results = opt_results
                    st.session_state.optimization_done = True
                    st.session_state.processing_history.append("Optimization completed")
                    
                    st.success("Optimization completed successfully!")
                    
                    st.subheader("Optimization Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Optimal Start Time (t1)",
                            f"{opt_results['optimal_t_start']:.2f} s",
                            delta=f"{opt_results['optimal_t_start_mins']:.2f} mins"
                        )
                    with col2:
                        st.metric(
                            "Optimal End Time (t2)",
                            f"{opt_results['optimal_t_end']:.2f} s",
                            delta=f"{opt_results['optimal_t_end_mins']:.2f} mins"
                        )
                    
                    with st.expander("View Full Optimization Results"):
                        st.json(opt_results)
                
                else:
                    st.error(f"Optimization failed: {opt_results.get('error', 'Unknown error')}")
        
        else:
            st.subheader("Manual Input Mode")
            st.info("Enter optimization results manually if available")
            
            col1, col2 = st.columns(2)
            with col1:
                manual_t1 = st.number_input("Optimal Start Time t1 (seconds)", value=5000.0, min_value=0.0)
            with col2:
                manual_t2 = st.number_input("Optimal End Time t2 (seconds)", value=6000.0, min_value=0.0)
            
            if st.button("Use Manual Input"):
                st.session_state.optimization_results = {
                    'success': True,
                    'optimal_t_start': manual_t1,
                    'optimal_t_end': manual_t2,
                    'optimal_t_start_mins': manual_t1 / 60,
                    'optimal_t_end_mins': manual_t2 / 60,
                    'manual_input': True
                }
                st.session_state.optimization_done = True
                st.session_state.processing_history.append("Manual optimization values entered")
                st.success("Manual values saved!")
        
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("Back to Peak Integration"):
                st.session_state.current_step = 3
                st.rerun()
        
        with col2:
            if st.session_state.optimization_done:
                if st.button("Calculate Timing Parameters", type="primary"):
                    st.session_state.current_step = 5
                    st.rerun()
        
        with col3:
            if st.button("Skip to Export"):
                st.session_state.current_step = 6
                st.rerun()
    
    elif st.session_state.current_step == 5:
        st.header("Step 5: Timing Parameters Calculation")
        
        if not st.session_state.optimization_done:
            st.error("Please complete optimization first or enter manual values")
            if st.button("Go to Optimization"):
                st.session_state.current_step = 4
                st.rerun()
            return
        
        opt_results = st.session_state.optimization_results
        
        st.subheader("Optimization Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Start Time (t1)", f"{opt_results['optimal_t_start']:.2f} s")
        with col2:
            st.metric("End Time (t2)", f"{opt_results['optimal_t_end']:.2f} s")
        
        st.markdown("---")
        
        if st.button("Calculate Timing Parameters", type="primary", use_container_width=True):
            calc = st.session_state.opt_calculator
            
            timing_params = calc.calculate_timing_parameters(
                opt_results['optimal_t_start'],
                opt_results['optimal_t_end']
            )
            
            st.session_state.timing_parameters = timing_params
            st.session_state.processing_history.append("Timing parameters calculated")
            
            st.success("Timing parameters calculated!")
            
            st.subheader("Calculated Timing Parameters")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Non-Pooled Time 1",
                    f"{timing_params['nonpooled_time_1']:.2f} s",
                    delta=f"{timing_params['nonpooled_time_1_mins']:.2f} mins"
                )
            
            with col2:
                st.metric(
                    "Pooling Time",
                    f"{timing_params['pooling_time']:.2f} s",
                    delta=f"{timing_params['pooling_time_mins']:.2f} mins"
                )
            
            with col3:
                st.metric(
                    "Non-Pooled Time 2",
                    f"{timing_params['nonpooled_time_2']:.2f} s",
                    delta=f"{timing_params['nonpooled_time_2_mins']:.2f} mins"
                )
            
            with st.expander("Configuration Details"):
                st.write(f"Baseline: {timing_params['baseline']} seconds")
                st.write(f"Total Time: {timing_params['total_time']} seconds")
                st.write(f"t1 (Start): {timing_params['t1']:.2f} seconds")
                st.write(f"t2 (End): {timing_params['t2']:.2f} seconds")
            
            timing_text = f"""# Timing parameters calculated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# From optimization: t1={timing_params['t1']:.2f}s, t2={timing_params['t2']:.2f}s
# User: {st.session_state.user_name}
# File: {st.session_state.hplc_file_name}

{timing_params['nonpooled_time_1_mins']:.2f}
{timing_params['pooling_time_mins']:.2f}
{timing_params['nonpooled_time_2_mins']:.2f}
"""
            
            st.session_state.timing_parameters_text = timing_text
            
            st.download_button(
                "Download Timing Parameters File",
                timing_text,
                "timing_parameters.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        st.markdown("---")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("Back to Optimization"):
                st.session_state.current_step = 4
                st.rerun()
        
        with col2:
            if st.button("Proceed to Export", type="primary"):
                st.session_state.current_step = 6
                st.rerun()
    
    elif st.session_state.current_step == 6:
        st.header("Step 6: Export Results")
        
        if not st.session_state.processing_done:
            st.error("Please complete peak integration first")
            if st.button("Go to Peak Integration"):
                st.session_state.current_step = 3
                st.rerun()
            return
        
        st.success("All processing completed!")
        
        st.subheader("Processing Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("User Information")
            st.write(f"Name: {st.session_state.user_name}")
            st.write(f"File: {st.session_state.hplc_file_name}")
            st.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        with col2:
            st.info("Processing Steps Completed")
            if st.session_state.processing_done:
                st.write("Peak Integration: Complete")
            if st.session_state.optimization_done:
                st.write("Optimization: Complete")
            if 'timing_parameters' in st.session_state:
                st.write("Timing Parameters: Complete")
        
        with st.expander("Full Processing History"):
            for i, item in enumerate(st.session_state.processing_history, 1):
                st.write(f"{i}. {item}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        user_name_clean = re.sub(r'[^a-zA-Z0-9_]', '_', st.session_state.user_name)
        
        export_data = {
            "user_info": {
                "user_name": st.session_state.user_name,
                "creation_date": datetime.now().isoformat(),
                "user_id": user_name_clean
            },
            "file_info": {
                "filename": st.session_state.hplc_file_name,
                "processed_date": timestamp
            },
            "peak_integration_results": st.session_state.hplc_results,
            "processing_history": st.session_state.processing_history
        }
        
        if st.session_state.optimization_done:
            export_data["optimization_results"] = st.session_state.optimization_results
        
        if 'timing_parameters' in st.session_state:
            export_data["timing_parameters"] = st.session_state.timing_parameters
        
        json_filename = f"hplc_results_{user_name_clean}_{timestamp}.json"
        save_parameters_to_json(export_data, json_filename)
        
        st.subheader("Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("Complete Results (JSON)")
            with open(json_filename, 'r') as f:
                json_content = f.read()
            st.download_button(
                "Download Complete Results",
                json_content,
                json_filename,
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            if 'timing_parameters_text' in st.session_state:
                st.markdown("Timing Parameters (TXT)")
                st.download_button(
                    "Download Timing Parameters",
                    st.session_state.timing_parameters_text,
                    f"timing_parameters_{user_name_clean}_{timestamp}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        
        st.markdown("Peak Integration Results (CSV)")
        
        results = st.session_state.hplc_results
        results_df = pd.DataFrame({
            'Peak': ['Acid', 'Main', 'Base1', 'Base2', 'Total'],
            'Area': [
                results['peak_areas']['acid'],
                results['peak_areas']['main'],
                results['peak_areas']['base1'],
                results['peak_areas']['base2'],
                results['peak_areas']['total']
            ],
            'Percentage': [
                results['peak_percentages']['acid_pct'],
                results['peak_percentages']['main_pct'],
                results['peak_percentages']['base1_pct'],
                results['peak_percentages']['base2_pct'],
                100.0
            ]
        })
        
        csv_filename = f"peak_results_{user_name_clean}_{timestamp}.csv"
        csv_content = results_df.to_csv(index=False)
        
        st.download_button(
            "Download Peak Results CSV",
            csv_content,
            csv_filename,
            mime="text/csv",
            use_container_width=True
        )
        
        st.subheader("Final Results Summary")
        st.dataframe(results_df, use_container_width=True)
        
        st.markdown("---")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("Back to Timing Parameters"):
                if 'timing_parameters' in st.session_state:
                    st.session_state.current_step = 5
                else:
                    st.session_state.current_step = 4
                st.rerun()
        
        with col2:
            if st.button("Start New Processing", type="primary"):
                user_name = st.session_state.user_name
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.session_state.user_name = user_name
                st.session_state.current_step = 2
                st.session_state.hplc_processor = HPLCDataProcessor()
                st.session_state.opt_calculator = OptimizationCalculator()
                st.rerun()


main()

from chatbot import render_chatbot
render_chatbot("06_HPLC")

