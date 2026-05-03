import streamlit as st
import os

st.set_page_config(
    page_title="SpectraTwin",
    page_icon="🔬",
    layout="wide"
)

# Display logo at the top
logo_path = os.path.join(os.path.dirname(__file__), "Logo", "Spectra.png")
if os.path.exists(logo_path):
    col_logo = st.columns([1, 1, 1])[1]
    with col_logo:
        st.image(logo_path, width=250)

if 'user_name' not in st.session_state or st.session_state.user_name == "":
    st.title("SpectraTwin")
    st.markdown("### Welcome! Please enter your name to continue")
    
    user_name = st.text_input("Your Name:")
    
    if st.button("Start"):
        if user_name:
            st.session_state.user_name = user_name
            st.rerun()
        else:
            st.error("Please enter your name")
else:
    # Minimal Centered Hero Section
    st.markdown(f"<h1 style='text-align: center;'>SpectraTwin</h1>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align: center; color: #6e6e73; margin-bottom: 2rem;'>Welcome, {st.session_state.user_name}!</h4>", unsafe_allow_html=True)

    st.info(
        "**Overview**: This app turns complex spectral preprocessing into a clear, repeatable workflow. "
        "Built for reliability, it focuses on cleaning spectra, resolving artifacts, and extracting stable features without the hassle."
    )
    
    st.write("---")

    # Clean, concise 2-column layout for features
    st.subheader("🛠️ Core Capabilities")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        #### 📥 Smart Intake & Preprocessing
        - **Robust Validation**: Early shape, type, and outlier detection.
        - **Data Integrity**: Target selection and safe leak-free train-test splits.
        - **Modality Support**: Optimized presets for FTIR, Raman, NIR, and Mass Spec.
        
        #### 🧹 Noise & Artifact Control
        - **Denoising**: Wavelet denoising and peak-preserving smoothing.
        - **Correction**: Powerful baseline corrections and scatter normalization.
        - **Feature Tuning**: OPLS and PCA for targeted covariance reduction.
        """)
        
    with col2:
        st.markdown("""
        #### 🔄 Reproducible Pipelines
        - **Tracking**: Complete preprocessing history and pipeline rollbacks.
        - **Augmentation**: Physics-preserving data generation.
        - **Transparency**: No hidden transformations or stale states.

        #### ⚡ Guided vs Automated
        - **Guided**: Step-by-step interactive workflow with live diagnostics (Steps 00-07).
        - **Automated**: One-click end-to-end processing (Step 08).
        - **Deployment**: Save manifests and predict safely with drift checks.
        """)

    st.write("---")

    # Emphasize the single click pipeline as a Call-To-Action
    st.markdown("### 🚀 Quick Start: One-Click Pipeline")
    st.success(
        "Want to skip the manual setup? Upload your data, pick your spectral type, and let the system handle **everything** "
        "(preprocessing, reduction, augmentation, and model training). \n\n"
        "👉 **Head to Step 08 · One Click Pipeline in the sidebar to start instantly.**"
    )

with st.sidebar:
    st.markdown("---")
    st.markdown("💡 **Powered by Spectroscopy AI**")

# Safely import and render chatbot
try:
    from chatbot import render_chatbot
    render_chatbot("Home Page")
except Exception as e:
    st.sidebar.info("💬 Chatbot feature requires additional dependencies.")
