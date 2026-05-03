import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve
import pickle
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots

## remove .pkl

class ExperimentalDesigner:
    def __init__(self, 
                 project_type: str, 
                 y_component_names: list,
                 process_factors: list = None):
        if project_type not in ['process', 'spectral']:
            raise ValueError("project_type must be 'process' or 'spectral'")
        
        self.project_type = project_type
        self.y_component_names = y_component_names
        self.n_y_outputs = len(y_component_names)
        
        self.X_observed = []
        self.y_observed = []
        self.y_scaler = StandardScaler()
        
        self.gpr_base_estimator = GaussianProcessRegressor(
            kernel=Matern(nu=2.5) + WhiteKernel(noise_level=0.1),
            alpha=1e-6,
            n_restarts_optimizer=10,
            random_state=42
        )
        self.gpr = MultiOutputRegressor(self.gpr_base_estimator)
        self.is_model_trained = False
        
        if self.project_type == 'process':
            if not process_factors:
                raise ValueError("process_factors required for 'process' projects")
            self.process_factors = process_factors
            self.factor_names = [f['name'] for f in self.process_factors]
            self.n_factors = len(self.process_factors)
            self.min_bounds = np.array([f['min'] for f in self.process_factors])
            self.max_bounds = np.array([f['max'] for f in self.process_factors])
            self.x_scaler = MinMaxScaler()
        
        elif self.project_type == 'spectral':
            self.pca = None
            self.n_pca_components = 15
            self.x_scaler = StandardScaler()
            self.spectral_pipeline = None

    def _scale_to_original(self, lhs_matrix, min_b, max_b):
        return min_b + lhs_matrix * (max_b - min_b)

    def _dict_to_array(self, x_dict):
        return np.array([x_dict[name] for name in self.factor_names])

    def _array_to_dict(self, x_array):
        return {name: round(val, 5) for name, val in zip(self.factor_names, x_array)}

    def generate_initial_lhs(self, n_samples):
        sampler = qmc.LatinHypercube(d=self.n_factors, seed=42)
        lhs_matrix_01 = sampler.random(n=n_samples)
        lhs_scaled = self._scale_to_original(lhs_matrix_01, self.min_bounds, self.max_bounds)
        experiments = [self._array_to_dict(row) for row in lhs_scaled]
        return experiments

    def _build_spectral_pipeline(self):
        n_samples = len(self.X_observed)
        n_features = self.X_observed[0].shape[0] if n_samples > 0 else 20
        n_comps = min(self.n_pca_components, n_samples - 1, n_features)
        if n_comps <= 0: 
            n_comps = 1
        self.pca = PCA(n_components=n_comps, random_state=42)
        self.spectral_pipeline = Pipeline([
            ('scaler', self.x_scaler),
            ('pca', self.pca),
            ('gpr', self.gpr)
        ])

    def register_experiment_result(self, x_input, y_dict):
        try:
            y_array = np.array([y_dict[name] for name in self.y_component_names])
        except KeyError as e:
            st.error(f"Missing component: {e}")
            return False
            
        if self.project_type == 'process':
            x_array = self._dict_to_array(x_input)
        else:
            if isinstance(x_input, pd.Series):
                x_array = x_input.values
            else:
                x_array = x_input
        
        for existing_x in self.X_observed:
            if np.allclose(x_array, existing_x):
                st.warning("Duplicate X entry skipped")
                return False

        self.X_observed.append(x_array)
        self.y_observed.append(y_array)
        self.is_model_trained = False
        return True

    def _train_gpr_model(self):
        if self.project_type == 'spectral':
            self._build_spectral_pipeline()
            X = np.array(self.X_observed)
            y = np.array(self.y_observed)
            y_scaled = self.y_scaler.fit_transform(y)
            
            try:
                self.spectral_pipeline.fit(X, y_scaled)
                self.is_model_trained = True
                return True
            except Exception as e:
                st.error(f"Training error: {e}")
                return False

        min_samples = max(self.n_factors + 1, 5)
        if len(self.y_observed) < min_samples:
            st.warning(f"Need {min_samples} samples, have {len(self.y_observed)}")
            return False

        y = np.array(self.y_observed)
        y_scaled = self.y_scaler.fit_transform(y)
        X = np.array(self.X_observed)
        X_scaled = self.x_scaler.fit_transform(X)
        
        try:
            self.gpr.fit(X_scaled, y_scaled)
            self.is_model_trained = True
            return True
        except Exception as e:
            st.error(f"Training error: {e}")
            return False

    def suggest_next_experiment(self, n_candidates=5000):
        if not self.is_model_trained:
            if not self._train_gpr_model():
                return None

        candidate_pool = np.random.uniform(
            low=self.min_bounds,
            high=self.max_bounds,
            size=(n_candidates, self.n_factors)
        )
        candidates_scaled = self.x_scaler.transform(candidate_pool)
        
        all_variances = []
        try:
            for estimator in self.gpr.estimators_:
                _, std_dev = estimator.predict(candidates_scaled, return_std=True)
                all_variances.append(std_dev**2)
            total_variance = np.sum(np.array(all_variances), axis=0)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None
            
        best_next_index = np.argmax(total_variance)
        best_next_experiment = candidate_pool[best_next_index]
        next_experiment = self._array_to_dict(best_next_experiment)
        return next_experiment, total_variance[best_next_index]

    def generate_y_space_plan(self, y_min_bounds, y_max_bounds, n_samples):
        sampler = qmc.LatinHypercube(d=self.n_y_outputs, seed=42)
        lhs_matrix_01 = sampler.random(n=n_samples)
        lhs_scaled = self._scale_to_original(lhs_matrix_01, y_min_bounds, y_max_bounds)
        return lhs_scaled

    def analyze_data_sufficiency(self):
        min_samples = max(self.n_y_outputs + 5, 10)
        if len(self.y_observed) < min_samples:
            st.error(f"Need at least {min_samples} samples for analysis")
            return None

        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        self._build_spectral_pipeline()
        
        n_points = min(10, len(self.y_observed) - 5)
        train_sizes_abs = np.linspace(min_samples, int(len(self.y_observed) * 0.8), n_points, dtype=int)

        try:
            with st.spinner("Running learning curve analysis"):
                train_sizes, train_scores, test_scores = learning_curve(
                    self.spectral_pipeline,
                    X, y,
                    train_sizes=train_sizes_abs,
                    cv=5,
                    scoring='neg_root_mean_squared_error',
                    n_jobs=-1,
                    random_state=42
                )
            
            train_scores_mean = np.mean(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            train_rmse = -train_scores_mean
            test_rmse = -test_scores_mean

            return {
                'train_sizes': train_sizes,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse
            }
        except Exception as e:
            st.error(f"Analysis error: {e}")
            return None

def load_data_file(uploaded_file):
    """Load data from CSV, Excel, or TXT file"""
    try:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        if file_ext == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_ext in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        elif file_ext == 'txt':
            df = pd.read_csv(uploaded_file, delimiter='\t')
        else:
            st.error(f"Unsupported file format: {file_ext}")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

if 'designer' not in st.session_state:
    st.session_state.designer = None
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'temp_data' not in st.session_state:
    st.session_state.temp_data = {}

def save_state(designer, filename):
    try:
        buffer = io.BytesIO()
        pickle.dump(designer, buffer)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"Error saving: {e}")
        return None

def load_state(uploaded_file):
    try:
        designer = pickle.load(uploaded_file)
        return designer
    except Exception as e:
        st.error(f"Error loading: {e}")
        return None

def home_page():
    st.title("🔬 Advanced Experimental Designer")
    st.markdown("### Design experiments and analyze data sufficiency")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Create New Project")
        if st.button("➕ New Project", use_container_width=True):
            st.session_state.page = 'new_project'
            st.rerun()
    
    with col2:
        st.subheader("Load Existing Project")
        uploaded_file = st.file_uploader("Upload project file (.pkl)", type=['pkl'])
        if uploaded_file:
            designer = load_state(uploaded_file)
            if designer:
                st.session_state.designer = designer
                st.session_state.page = 'project_menu'
                st.success(f"Loaded {designer.project_type} project with {len(designer.y_observed)} data points")
                st.rerun()

def new_project_page():
    st.title("Create New Project")
    
    project_type = st.selectbox("Project Type", ['process', 'spectral'])
    
    process_factors = None
    if project_type == 'process':
        st.subheader("Process Factors")
        if 'n_factors' not in st.session_state:
            st.session_state.n_factors = 3
        
        n_factors = st.number_input(
            "Number of factors", 
            min_value=1, 
            max_value=10, 
            value=st.session_state.n_factors,
            key='n_factors_input'
        )
        st.session_state.n_factors = n_factors
    
    with st.form("new_project_form"):
        st.subheader("Y Components (Outputs)")
        y_components = st.text_input("Component names (comma-separated)", "Component_A, Component_B")
        
        if project_type == 'process':
            factor_data = []
            for i in range(st.session_state.n_factors):
                st.markdown(f"**Factor {i+1}**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    name = st.text_input(f"Name", f"Factor_{i+1}", key=f"fname_{i}")
                with col2:
                    min_val = st.number_input(f"Min", value=0.0, key=f"fmin_{i}")
                with col3:
                    max_val = st.number_input(f"Max", value=100.0, key=f"fmax_{i}")
                factor_data.append({'name': name, 'min': min_val, 'max': max_val})
            
            process_factors = factor_data
        
        submitted = st.form_submit_button("Create Project", use_container_width=True)
        
        if submitted:
            y_names = [name.strip() for name in y_components.split(',') if name.strip()]
            if not y_names:
                st.error("Please provide at least one component name")
            else:
                try:
                    designer = ExperimentalDesigner(
                        project_type=project_type,
                        y_component_names=y_names,
                        process_factors=process_factors
                    )
                    st.session_state.designer = designer
                    st.session_state.page = 'project_menu'
                    # Clear the n_factors state when project is created
                    if 'n_factors' in st.session_state:
                        del st.session_state.n_factors
                    st.success("Project created successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    
    if st.button("← Back to Home"):
        st.session_state.page = 'home'
        # Clear n_factors when going back
        if 'n_factors' in st.session_state:
            del st.session_state.n_factors
        st.rerun()

def project_menu_page():
    designer = st.session_state.designer
    
    st.title(f"  {designer.project_type.title()} Project")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Data Points", len(designer.y_observed))
    with col2:
        st.metric("Components", designer.n_y_outputs)
    with col3:
        st.metric("Model Trained", "✓" if designer.is_model_trained else "✗")
    with col4:
        if st.button(" Save Project"):
            buffer = save_state(designer, "project.pkl")
            if buffer:
                st.download_button(
                    "Download Project File",
                    data=buffer,
                    file_name="experimental_design_project.pkl",
                    mime="application/octet-stream"
                )
    
    st.divider()
    
    tabs = st.tabs(["  Data Management", "🔬 Design & Analysis", "  Diagnostics"])
    
    with tabs[0]:
        data_management_tab(designer)
    
    with tabs[1]:
        design_analysis_tab(designer)
    
    with tabs[2]:
        diagnostics_tab(designer)
    
    st.divider()
    if st.button("← Back to Home"):
        st.session_state.page = 'home'
        st.rerun()

def data_management_tab(designer):
    st.subheader("Data Management")
    
    upload_col, manual_col = st.columns(2)
    
    with upload_col:
        st.markdown("#### Upload Data File")
        st.info("Supports: CSV, Excel (.xlsx, .xls), TXT (tab-delimited)")
        
        uploaded_file = st.file_uploader(
            "Upload data file", 
            type=['csv', 'xlsx', 'xls', 'txt'],
            key='data_upload'
        )
        
        if uploaded_file:
            df = load_data_file(uploaded_file)
            
            if df is not None:
                st.success(f"Loaded file: {uploaded_file.name}")
                st.dataframe(df.head(10))
                
                st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
                
                # TRY BOTH FORMATS: y_componentname AND componentname
                y_col_names_with_prefix = [f"y_{name}" for name in designer.y_component_names]
                y_col_names_without_prefix = designer.y_component_names
                
                # Check which format exists in the dataframe
                missing_with_prefix = [col for col in y_col_names_with_prefix if col not in df.columns]
                missing_without_prefix = [col for col in y_col_names_without_prefix if col not in df.columns]
                
                # Determine which format to use
                if len(missing_without_prefix) == 0:
                    # Use column names without prefix
                    y_col_names = y_col_names_without_prefix
                    use_prefix = False
                elif len(missing_with_prefix) == 0:
                    # Use column names with prefix
                    y_col_names = y_col_names_with_prefix
                    use_prefix = True
                else:
                    # Neither format works completely
                    st.error(f"Missing required columns!")
                    st.error(f"Looking for either: {', '.join(y_col_names_with_prefix)}")
                    st.error(f"Or: {', '.join(y_col_names_without_prefix)}")
                    st.info(f"Available columns: {', '.join(df.columns.tolist())}")
                    y_col_names = None
                
                if y_col_names:
                    st.success(f"✓ Found Y columns: {', '.join(y_col_names)}")
                    
                    if st.button("Load Data from File", key='load_file_btn'):
                        rows_processed = 0
                        rows_skipped = 0
                        
                        with st.spinner("Processing data..."):
                            for _, row in df.iterrows():
                                try:
                                    if row[y_col_names].isnull().values.any():
                                        rows_skipped += 1
                                        continue
                                    
                                    # Build y_dict using the correct column names
                                    y_dict = {}
                                    for i, name in enumerate(designer.y_component_names):
                                        y_dict[name] = float(row[y_col_names[i]])
                                    
                                    if designer.project_type == 'process':
                                        x_input = {name: float(row[name]) for name in designer.factor_names}
                                    else:
                                        # For spectral: exclude y columns
                                        x_cols = [c for c in df.columns if c not in y_col_names]
                                        x_input = row[x_cols].values
                                    
                                    if designer.register_experiment_result(x_input, y_dict):
                                        rows_processed += 1
                                except Exception as e:
                                    rows_skipped += 1
                                    continue
                        
                        st.success(f"✓ Loaded {rows_processed} data points")
                        if rows_skipped > 0:
                            st.warning(f"⚠ Skipped {rows_skipped} rows (invalid/duplicate data)")
                        st.rerun()
    
    with manual_col:
        st.markdown("#### Manual Entry")
        if designer.project_type == 'process':
            with st.form("manual_entry"):
                st.markdown("**Factor Values**")
                x_dict = {}
                for factor in designer.process_factors:
                    x_dict[factor['name']] = st.number_input(
                        factor['name'],
                        value=(factor['min'] + factor['max']) / 2,
                        min_value=factor['min'],
                        max_value=factor['max'],
                        key=f"manual_{factor['name']}"
                    )
                
                st.markdown("**Component Values**")
                y_dict = {}
                for name in designer.y_component_names:
                    y_dict[name] = st.number_input(f"{name}", value=0.0, key=f"manual_y_{name}")
                
                if st.form_submit_button("Add Data Point", use_container_width=True):
                    if designer.register_experiment_result(x_dict, y_dict):
                        st.success("✓ Data point added!")
                        st.rerun()
        else:
            st.info("Spectral mode: Please use file upload (CSV/Excel/TXT)")
    
    if len(designer.y_observed) > 0:
        st.divider()
        st.markdown("#### Current Dataset")
        
        if designer.project_type == 'process':
            data_dict = {}
            for i, x in enumerate(designer.X_observed):
                x_dict = designer._array_to_dict(x)
                for key, val in x_dict.items():
                    if key not in data_dict:
                        data_dict[key] = []
                    data_dict[key].append(val)
            
            for j, name in enumerate(designer.y_component_names):
                data_dict[name] = [y[j] for y in designer.y_observed]
            
            df = pd.DataFrame(data_dict)
            st.dataframe(df, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                csv = df.to_csv(index=False)
                st.download_button(" CSV", csv, "data.csv", "text/csv", use_container_width=True)
            with col2:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='Data')
                excel_data = buffer.getvalue()
                st.download_button("  Excel", excel_data, "data.xlsx", 
                                 "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                 use_container_width=True)
            with col3:
                txt = df.to_csv(index=False, sep='\t')
                st.download_button("  TXT", txt, "data.txt", "text/plain", use_container_width=True)
        else:
            st.info(f"Spectral data: {len(designer.y_observed)} samples loaded")
            
            
def design_analysis_tab(designer):
    st.subheader("Design & Analysis")
    
    if designer.project_type == 'process':
        process_design_section(designer)
    else:
        spectral_analysis_section(designer)

def process_design_section(designer):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Generate LHS Experiments")
        default_n = max(designer.n_factors + 5, 10)
        n_samples = st.number_input("Number of samples", min_value=5, value=default_n, key='lhs_n')
        
        if st.button("Generate LHS Design", use_container_width=True):
            experiments = designer.generate_initial_lhs(n_samples)
            
            exp_data = []
            for exp in experiments:
                row = exp.copy()
                for name in designer.y_component_names:
                    row[f"y_{name}"] = ""
                exp_data.append(row)
            
            df = pd.DataFrame(exp_data)
            st.dataframe(df, use_container_width=True)
            
            st.markdown("**Download Experiment Plan:**")
            dcol1, dcol2, dcol3 = st.columns(3)
            
            with dcol1:
                csv = df.to_csv(index=False)
                st.download_button("  CSV", csv, "lhs_plan.csv", "text/csv", use_container_width=True)
            
            with dcol2:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='Experiments')
                excel_data = buffer.getvalue()
                st.download_button("  Excel", excel_data, "lhs_plan.xlsx", 
                                 "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                 use_container_width=True)
            
            with dcol3:
                txt = df.to_csv(index=False, sep='\t')
                st.download_button("  TXT", txt, "lhs_plan.txt", "text/plain", use_container_width=True)
    
    with col2:
        st.markdown("#### Bayesian Optimization")
        min_required = max(designer.n_factors + 1, 5)
        
        if len(designer.y_observed) < min_required:
            st.warning(f"Need at least {min_required} data points")
            st.info(f"Current: {len(designer.y_observed)} points")
        else:
            n_candidates = st.number_input(
                "Candidate pool size", 
                min_value=1000, 
                max_value=20000, 
                value=5000,
                key='bayes_candidates'
            )
            
            if st.button("🔍 Suggest Next Experiment", use_container_width=True):
                with st.spinner("Searching for optimal point..."):
                    result = designer.suggest_next_experiment(n_candidates)
                
                if result:
                    next_exp, variance = result
                    st.success("✓ Next experiment suggested!")
                    
                    st.markdown("**Suggested Parameters:**")
                    exp_df = pd.DataFrame([next_exp])
                    st.dataframe(exp_df, use_container_width=True)
                    
                    st.info(f"Total Uncertainty: {variance:.4f}")
                    
                    st.markdown("**Download Suggestion:**")
                    csv = exp_df.to_csv(index=False)
                    st.download_button("  CSV", csv, "next_experiment.csv", "text/csv")

def spectral_analysis_section(designer):
    tab1, tab2 = st.tabs(["Y-Space Design", "Learning Curve"])
    
    with tab1:
        st.markdown("#### Generate Y-Space Experimental Plan")
        
        st.write("Define target ranges for each component:")
        y_min = []
        y_max = []
        
        for name in designer.y_component_names:
            col1, col2 = st.columns(2)
            with col1:
                min_val = st.number_input(f"Min {name}", value=0.0, key=f"ymin_{name}")
            with col2:
                max_val = st.number_input(f"Max {name}", value=100.0, key=f"ymax_{name}")
            y_min.append(min_val)
            y_max.append(max_val)
        
        base_samples = 20
        samples_per_component = 10 * designer.n_y_outputs
        samples_for_pca = designer.n_pca_components * 2
        total_guess = base_samples + samples_per_component + samples_for_pca
        
        st.info(f"💡 Suggested minimum samples: **{total_guess}**")
        
        n_samples = st.number_input("Number of samples to plan", min_value=5, value=total_guess, key='yspace_n')
        
        if st.button("Generate Y-Space Plan", use_container_width=True):
            y_min_bounds = np.array(y_min)
            y_max_bounds = np.array(y_max)
            plan = designer.generate_y_space_plan(y_min_bounds, y_max_bounds, n_samples)
            
            headers = [f"target_y_{name}" for name in designer.y_component_names]
            df = pd.DataFrame(plan, columns=headers)
            st.dataframe(df, use_container_width=True)
            
            st.markdown("**Download Y-Space Plan:**")
            dcol1, dcol2, dcol3 = st.columns(3)
            
            with dcol1:
                csv = df.to_csv(index=False)
                st.download_button("  CSV", csv, "y_space_plan.csv", "text/csv", use_container_width=True)
            
            with dcol2:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='Y-Space Plan')
                excel_data = buffer.getvalue()
                st.download_button("  Excel", excel_data, "y_space_plan.xlsx", 
                                 "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                 use_container_width=True)
            
            with dcol3:
                txt = df.to_csv(index=False, sep='\t')
                st.download_button("  TXT", txt, "y_space_plan.txt", "text/plain", use_container_width=True)
    
    with tab2:
        st.markdown("#### Learning Curve Analysis")
        
        min_samples = max(designer.n_y_outputs + 5, 10)
        if len(designer.y_observed) < min_samples:
            st.warning(f"⚠ Need at least {min_samples} data points for analysis")
            st.info(f"Current: {len(designer.y_observed)} points")
        else:
            if st.button("🔍 Run Learning Curve Analysis", use_container_width=True):
                results = designer.analyze_data_sufficiency()
                if results:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=results['train_sizes'],
                        y=results['train_rmse'],
                        mode='lines+markers',
                        name='Training RMSE',
                        line=dict(color='red', width=3),
                        marker=dict(size=8)
                    ))
                    fig.add_trace(go.Scatter(
                        x=results['train_sizes'],
                        y=results['test_rmse'],
                        mode='lines+markers',
                        name='Cross-Validation RMSE',
                        line=dict(color='green', width=3),
                        marker=dict(size=8)
                    ))
                    fig.update_layout(
                        title='Learning Curve: Data Sufficiency Analysis',
                        xaxis_title='Number of Training Samples',
                        yaxis_title='RMSE (Average across components)',
                        height=500,
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    
                    final_rmse = results['test_rmse'][-1]
                    initial_rmse = results['test_rmse'][0]
                    slope = (results['test_rmse'][-1] - results['test_rmse'][-2]) / results['test_rmse'][-2] if len(results['test_rmse']) > 1 else 0
                    
                    with col1:
                        st.metric("Final CV RMSE", f"{final_rmse:.4f}")
                        st.metric("Initial CV RMSE", f"{initial_rmse:.4f}")
                    
                    with col2:
                        improvement = ((initial_rmse - final_rmse) / initial_rmse * 100)
                        st.metric("Total Improvement", f"{improvement:.1f}%")
                        st.metric("Recent Slope", f"{slope*100:.2f}%")
                    
                    st.divider()
                    
                    if slope > -0.05:
                        st.success("**Interpretation: Curve is FLAT (Plateaued)**")
                        st.info("You likely have **ENOUGH DATA** for this model. Adding more similar samples may not improve performance significantly.")
                    else:
                        st.warning(" **Interpretation: Curve is STEEP (Decreasing)**")
                        st.info("Your model is **DATA-HUNGRY** and would benefit from more samples. Consider collecting additional data.")

def diagnostics_tab(designer):
    st.subheader("Model Diagnostics")
    
    if len(designer.y_observed) == 0:
        st.info("No data available yet. Add data to see diagnostics.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Project Information")
        info_data = {
            "Property": ["Project Type", "Data Points", "Components", "Model Trained"],
            "Value": [
                designer.project_type.title(),
                len(designer.y_observed),
                ', '.join(designer.y_component_names),
                "✓ Yes" if designer.is_model_trained else "✗ No"
            ]
        }
        st.dataframe(pd.DataFrame(info_data), use_container_width=True, hide_index=True)
        
        if designer.project_type == 'process':
            st.markdown("#### Factor Information")
            factor_data = {
                "Factor": designer.factor_names,
                "Min": [f['min'] for f in designer.process_factors],
                "Max": [f['max'] for f in designer.process_factors]
            }
            st.dataframe(pd.DataFrame(factor_data), use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### Component Statistics")
        y_data = np.array(designer.y_observed)
        stats_df = pd.DataFrame({
            'Component': designer.y_component_names,
            'Mean': np.round(y_data.mean(axis=0), 4),
            'Std': np.round(y_data.std(axis=0), 4),
            'Min': np.round(y_data.min(axis=0), 4),
            'Max': np.round(y_data.max(axis=0), 4)
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    if len(designer.y_observed) > 1:
        st.divider()
        st.markdown("#### Component Distributions")
        
        fig = make_subplots(
            rows=1, 
            cols=min(3, designer.n_y_outputs),
            subplot_titles=designer.y_component_names[:3]
        )
        
        for i, name in enumerate(designer.y_component_names[:3]):
            col_idx = i + 1
            y_vals = y_data[:, i]
            
            fig.add_trace(
                go.Histogram(x=y_vals, name=name, showlegend=False, nbinsx=20),
                row=1, col=col_idx
            )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    if designer.is_model_trained or designer._train_gpr_model():
        st.divider()
        st.markdown("#### Model Performance")
        
        if designer.project_type == 'spectral' and designer.spectral_pipeline:
            pca = designer.spectral_pipeline.named_steps['pca']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("PCA Components", pca.n_components_)
            with col2:
                st.metric("Variance Explained", f"{np.sum(pca.explained_variance_ratio_)*100:.2f}%")
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(range(1, pca.n_components_+1)),
                y=pca.explained_variance_ratio_,
                name='Explained Variance',
                marker_color='steelblue'
            ))
            fig.add_trace(go.Scatter(
                x=list(range(1, pca.n_components_+1)),
                y=np.cumsum(pca.explained_variance_ratio_),
                name='Cumulative',
                mode='lines+markers',
                yaxis='y2',
                marker=dict(color='red', size=8),
                line=dict(color='red', width=2)
            ))
            fig.update_layout(
                title='PCA Explained Variance',
                xaxis_title='Principal Component',
                yaxis_title='Variance Ratio',
                yaxis2=dict(title='Cumulative Variance', overlaying='y', side='right'),
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

if st.session_state.page == 'home':
    home_page()
elif st.session_state.page == 'new_project':
    new_project_page()
elif st.session_state.page == 'project_menu':
    project_menu_page()

from chatbot import render_chatbot
render_chatbot("00_Experimental design ")

