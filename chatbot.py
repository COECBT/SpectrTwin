import streamlit as st
import os

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

def init_chatbot_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "groq_api_key" not in st.session_state:
        # Try multiple sources for API key:
        # 1. Streamlit secrets (for Cloud deployment)
        # 2. Environment variables
        # 3. Default empty
        try:
            st.session_state.groq_api_key = st.secrets.get("GROQ_API_KEY", "")
        except:
            st.session_state.groq_api_key = os.getenv("GROQ_API_KEY", "")

PAGE_EXPLANATIONS = {
    "Home Page": "This is the main landing page. It provides an overview of the Spectroscopy ML Pipeline, its features, and access to all modules.",
    "00_Experimental design ": "Use this page to set up your experimental layout, input sample metadata, and organize the groups for your spectral analysis.",
    "01_Data_Visualization": "Upload your raw spectral data here to inspect it visually. You can look at individual spectra, check for obvious outliers, and understand the general shape of your signals.",
    "02_Preprocessing": "Crucial step: Apply baseline corrections, smoothing (like Savitzky-Golay), and normalization (like SNV or MSC) to remove noise and physical artifacts from your spectra.",
    "03_Model_Training": "Train supervised machine learning models (like PLS or classification algorithms) on your preprocessed data. Tune hyperparameters and view cross-validation scores.",
    "04_Control chart": "Monitor your predictions or spectral quality over time. Useful for catching instrumental drift or out-of-specification samples.",
    "05_Full_Pipeline": "A continuous, customizable workflow taking you from raw data to a fully trained model in sequential, verifiable steps.",
    "06_HPLC": "Dedicated tools for handling High-Performance Liquid Chromatography (HPLC) files, focusing on chromatogram peak alignment and integration.",
    "07_Model_prediction": "Load your already-trained models and fresh data to predict new outcomes instantly.",
    "08_One_Click_Pipeline": "The fully automated route. Just upload raw spectra, specify the modality, and let the system automatically preprocess, split, and train the best model."
}

def render_chatbot(page_context: str):
    if not GROQ_AVAILABLE:
        st.sidebar.info("💬 AI Chatbot: Groq module not installed. Install with: `pip install groq`")
        return
    
    init_chatbot_state()
    
    # CSS to make the Popover float at the bottom right
    st.markdown(
        """
        <style>
        [data-testid="stPopover"] {
            position: fixed;
            bottom: 30px;
            right: 30px;
            z-index: 9999;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Render the popover (a native Streamlit feature acting as a floating dialog)
    with st.popover("💬 AI Assistant"):
        st.markdown(f"**Current Page:** {page_context}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📖 Guide", use_container_width=True):
                explanation = PAGE_EXPLANATIONS.get(page_context, f"This is the {page_context} module. Follow the on-screen widgets to proceed.")
                st.session_state.messages.append({"role": "assistant", "content": f"**Page Guide:**\n\n{explanation}"})
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        if not st.session_state.groq_api_key:
            st.warning("Please enter your Groq API Key to enable the assistant.")
            st.markdown("[Get your free Groq API key here](https://console.groq.com/keys)")
            api_key = st.text_input("Groq API Key (Free)", type="password", key="groq_key_input")
            if st.button("Save Key"):
                st.session_state.groq_api_key = api_key
                st.rerun()
            return
            
        # Display chat messages with a specific height to keep the popover tidy
        chat_container = st.container(height=400)
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    
        # Input for the user
        prompt = st.chat_input("Ask a question about this step...")
        
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            try:
                if not GROQ_AVAILABLE:
                    raise ImportError("Groq module not available. Please install it with: pip install groq")
                    
                client = Groq(api_key=st.session_state.groq_api_key)
                
                system_prompt = (
                    f"You are a helpful AI assistant integrated into a Spectroscopy Machine Learning Pipeline application. "
                    f"The user is currently viewing the page: '{page_context}'. "
                    f"Provide concise, helpful guidance, data interpretation tips, or Streamlit instructions based on this context. "
                    f"If they ask about preprocessing, PCA, PLS, or Raman/FTIR spectroscopy, answer as an expert."
                )
                
                # Prepare message history for the API
                api_messages = [{"role": "system", "content": system_prompt}]
                for m in st.session_state.messages:
                    api_messages.append({"role": m["role"], "content": m["content"]})
                    
                completion = client.chat.completions.create(
                    model="llama-3.1-8b-instant",  # Updated to the new active model
                    messages=api_messages,
                    temperature=0.7,
                    max_tokens=1024,
                )
                
                response_text = completion.choices[0].message.content
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.rerun()  # Rerun to correctly display the updated conversation in the container
            except Exception as e:
                st.error(f"Error communicating with Groq: {str(e)}")
                # Pop the user message if it failed so they can try again.
                st.session_state.messages.pop()
