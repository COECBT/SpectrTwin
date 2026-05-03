import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import random
from sklearn.model_selection import train_test_split

# Setup Page
st.set_page_config(page_title="NN Builder", page_icon="🧠", layout="wide")

st.title("Neural Network Builder")
st.markdown("Build Deep Neural Networks (DNN) or 1D Convolutional Neural Networks (1D-CNN).")

# Attempt TensorFlow Import safely
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten, Input
    from tensorflow.keras.callbacks import Callback
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    st.error("TensorFlow is not installed in your environment! Please run `pip install tensorflow` in your terminal to use this builder.")
    st.stop()

# -------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------------------------------------
if 'nn_layers' not in st.session_state:
    st.session_state.nn_layers = []
    
if 'trained_model_bytes' not in st.session_state:
    st.session_state.trained_model_bytes = None

if 'training_history' not in st.session_state:
    st.session_state.training_history = []

# Keras Callback for live Streamlit plotting!
class StreamlitLivePlot(Callback):
    def __init__(self, placeholder):
        super().__init__()
        self.placeholder = placeholder
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss', 0))
        val_loss = logs.get('val_loss', None)
        
        plot_data = {'Training Loss': self.losses}
        if val_loss is not None:
            self.val_losses.append(val_loss)
            plot_data['Validation Loss'] = self.val_losses
            
        df_plot = pd.DataFrame(plot_data)
        self.placeholder.line_chart(df_plot)

# -------------------------------------------------------------
# DATA INGESTION
# -------------------------------------------------------------
st.header("1. Select Dataset")
data_source = st.radio("Choose Data Source", ["Upload CSV", "Use Session State (Preprocessed)"], horizontal=True)

df = None
if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload Spectral Data (CSV)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("CSV Loaded Successfully!")
elif data_source == "Use Session State (Preprocessed)":
    if 'preprocessed_data' in st.session_state and st.session_state.preprocessed_data is not None:
        df = st.session_state.preprocessed_data
        st.success("Session State Data Linked!")
    else:
        st.warning("No preprocessed data found in session state. Please upload a CSV instead.")

target_col = None
X, y = None, None
if df is not None:
    target_col = st.selectbox("Select Target Variable (Y)", df.columns)
    if target_col:
        X = df.drop(columns=[target_col]).values
        y = df[target_col].values
        # Drop NaNs
        valid_idx = ~np.isnan(y)
        X = X[valid_idx]
        y = y[valid_idx]
        st.write(f"Data Shape: **{X.shape}** samples | Target Shape: **{y.shape}**")

st.markdown("---")

tab1, tab2 = st.tabs(["Visual Block Builder (Manual)", "AutoML Tuner (Automated)"])

with tab1:
    # -------------------------------------------------------------
    # BLOCK BUILDER UI
    # -------------------------------------------------------------
    st.header("2. Architecture Builder")
    st.markdown("Add layers sequentially to build your model. An Input Layer will be automatically mapped to your data shape.")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Current Network Stack")
        if not st.session_state.nn_layers:
            st.info("Your network is empty. Add a layer from the panel on the right.")
            
        for i, layer in enumerate(st.session_state.nn_layers):
            with st.container():
                # Card UI
                st.markdown(f"**Layer {i+1}: {layer['type']}**")
                
                c1, c2, c3, c4 = st.columns([4, 1, 1, 1])
                with c1:
                    if layer['type'] == 'Dense':
                        st.caption(f"Units: {layer['units']} | Activation: {layer['activation']}")
                    elif layer['type'] == 'Conv1D':
                        st.caption(f"Filters: {layer['filters']} | Kernel Size: {layer['kernel_size']} | Activation: {layer['activation']}")
                    elif layer['type'] == 'MaxPooling1D':
                        st.caption(f"Pool Size: {layer['pool_size']}")
                    elif layer['type'] == 'Dropout':
                        st.caption(f"Rate: {layer['rate']}")
                    elif layer['type'] == 'Flatten':
                        st.caption("Flattens multi-dimensional arrays natively.")
                
                with c2:
                    if st.button("▲ Up", key=f"up_{i}") and i > 0:
                        st.session_state.nn_layers.insert(i-1, st.session_state.nn_layers.pop(i))
                        st.rerun()
                with c3:
                    if st.button("▼ Down", key=f"down_{i}") and i < len(st.session_state.nn_layers)-1:
                        st.session_state.nn_layers.insert(i+1, st.session_state.nn_layers.pop(i))
                        st.rerun()
                with c4:
                    if st.button("🗑️ Del", key=f"del_{i}"):
                        st.session_state.nn_layers.pop(i)
                        st.rerun()
                st.markdown("---")

    with col2:
        st.subheader("Add New Layer")
        with st.form("add_layer_form"):
            layer_type = st.selectbox("Layer Type", ["Dense", "Conv1D", "MaxPooling1D", "Dropout", "Flatten"])
            
            units, filters, kernel_size, pool_size = 0, 0, 0, 0
            rate = 0.0
            activation = "relu"
            
            if layer_type == "Dense":
                units = st.number_input("Neurons (Units)", min_value=1, value=64, step=8)
                activation = st.selectbox("Activation", ["relu", "linear", "tanh", "sigmoid"])
            elif layer_type == "Conv1D":
                filters = st.number_input("Filters", min_value=1, value=32, step=8)
                kernel_size = st.number_input("Kernel Size", min_value=1, value=3, step=1)
                activation = st.selectbox("Activation", ["relu", "tanh"], key="conv_act")
            elif layer_type == "MaxPooling1D":
                pool_size = st.number_input("Pool Size", min_value=1, value=2, step=1)
            elif layer_type == "Dropout":
                rate = st.slider("Dropout Rate", min_value=0.0, max_value=0.9, value=0.2, step=0.05)
                
            submitted = st.form_submit_button("➕ Add Layer")
            if submitted:
                new_layer = {
                    'type': layer_type,
                    'units': units,
                    'filters': filters,
                    'kernel_size': kernel_size,
                    'pool_size': pool_size,
                    'rate': rate,
                    'activation': activation
                }
                st.session_state.nn_layers.append(new_layer)
                st.rerun()

    st.markdown("---")

    # -------------------------------------------------------------
    # TRAINING ENGINE
    # -------------------------------------------------------------
    st.header("3. Train & Export Model")

    t_col1, t_col2 = st.columns([1, 2])

    with t_col1:
        epochs = st.number_input("Epochs", min_value=1, value=50, step=10)
        batch_size = st.number_input("Batch Size", min_value=1, value=32, step=8)
        learning_rate = st.number_input("Learning Rate", value=0.001, format="%.4f")
        loss_fn = st.selectbox("Loss Function", ["mse", "mae"])
        optimizer = st.selectbox("Optimizer", ["adam", "rmsprop", "sgd"])
        
        train_clicked = st.button("Compile & Train Network", use_container_width=True, type="primary")

    with t_col2:
        loss_placeholder = st.empty()

    if train_clicked:
        if X is None or y is None:
            st.error("Please load and select a dataset and target variable first.")
        elif not st.session_state.nn_layers:
            st.error("Please add at least one layer to the network.")
        else:
            with st.spinner("Compiling and Training Neural Network..."):
                try:
                    has_conv = any(l['type'] in ['Conv1D', 'MaxPooling1D'] for l in st.session_state.nn_layers)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    if has_conv:
                        X_train = np.expand_dims(X_train, axis=2)
                        X_test = np.expand_dims(X_test, axis=2)
                        input_shape = (X_train.shape[1], 1)
                    else:
                        input_shape = (X_train.shape[1],)

                    model = Sequential()
                    model.add(Input(shape=input_shape))
                    
                    for layer in st.session_state.nn_layers:
                        if layer['type'] == 'Dense':
                            model.add(Dense(layer['units'], activation=layer['activation']))
                        elif layer['type'] == 'Conv1D':
                            model.add(Conv1D(layer['filters'], layer['kernel_size'], activation=layer['activation'], padding='same'))
                        elif layer['type'] == 'MaxPooling1D':
                            model.add(MaxPooling1D(layer['pool_size'], padding='same'))
                        elif layer['type'] == 'Dropout':
                            model.add(Dropout(layer['rate']))
                        elif layer['type'] == 'Flatten':
                            model.add(Flatten())
                    
                    model.add(Dense(1, activation='linear'))
                    
                    opt = tf.keras.optimizers.get(optimizer)
                    opt.learning_rate = learning_rate
                    model.compile(optimizer=opt, loss=loss_fn, metrics=['mae'])
                    
                    st_callback = StreamlitLivePlot(loss_placeholder)
                    model.fit(
                        X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[st_callback],
                        verbose=0
                    )
                    
                    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
                    st.success(f"Training Complete! Final Test Loss ({loss_fn.upper()}): {test_loss:.4f}")
                    
                    # Extract parameters for PKL compatibility
                    model_json = model.to_json()
                    model_weights = model.get_weights()
                    
                    payload = {
                        'architecture_json': model_json,
                        'weights': model_weights,
                        'model_type': 'keras_custom',
                        'input_features_count': X_train.shape[1]
                    }
                    
                    buffer = io.BytesIO()
                    pickle.dump(payload, buffer)
                    st.session_state.trained_model_bytes = buffer.getvalue()
                    
                except Exception as e:
                    st.error(f"Architecture Error: {str(e)}")
                    st.warning("Hint: Did you forget to add a 'Flatten' layer after Conv1D before placing Dense layers?")

with tab2:
    st.header("AutoML Tuner (Random Search)")
    st.markdown("Automatically search for the best neural network parameters without building layers manually.")
    
    a_col1, a_col2 = st.columns([1, 2])
    with a_col1:
        base_arch = st.selectbox("Base Architecture", ["DNN (Fully Connected)", "1D-CNN (Convolutional)"])
        num_trials = st.number_input("Number of Trials", min_value=2, max_value=50, value=10, step=2)
        auto_epochs = st.number_input("Epochs per Trial", min_value=10, value=30, step=10)
        auto_loss = st.selectbox("Loss Function ", ["mse", "mae"])
        auto_train = st.button("Start AutoML Tuning", use_container_width=True, type="primary")
        
    with a_col2:
        auto_status = st.empty()
        auto_progress = st.progress(0)
        best_metric_display = st.empty()
        
    if auto_train:
        if X is None or y is None:
            st.error("Please load and select a dataset and target variable first.")
        else:
            best_val_loss = float('inf')
            best_payload = None
            best_config = {}
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            is_cnn = "CNN" in base_arch
            if is_cnn:
                X_train_dl = np.expand_dims(X_train, axis=2)
                X_test_dl = np.expand_dims(X_test, axis=2)
                input_shape = (X_train.shape[1], 1)
            else:
                X_train_dl = X_train
                X_test_dl = X_test
                input_shape = (X_train.shape[1],)
                
            for trial in range(num_trials):
                auto_status.info(f"Running Trial {trial+1} / {num_trials}...")
                auto_progress.progress((trial) / num_trials)
                
                # Randomly sample hyperparameters
                num_layers = random.choice([1, 2, 3])
                units_choices = [32, 64, 128, 256]
                dropout_rate = random.choice([0.1, 0.2, 0.3, 0.4])
                lr = random.choice([0.01, 0.005, 0.001, 0.0005])
                
                model = Sequential()
                model.add(Input(shape=input_shape))
                
                if is_cnn:
                    for _ in range(num_layers):
                        model.add(Conv1D(random.choice([16, 32, 64]), kernel_size=3, activation='relu', padding='same'))
                        model.add(MaxPooling1D(2, padding='same'))
                    model.add(Flatten())
                    model.add(Dense(random.choice(units_choices), activation='relu'))
                else:
                    for _ in range(num_layers):
                        model.add(Dense(random.choice(units_choices), activation='relu'))
                        model.add(Dropout(dropout_rate))
                
                model.add(Dense(1, activation='linear'))
                
                opt = tf.keras.optimizers.Adam(learning_rate=lr)
                model.compile(optimizer=opt, loss=auto_loss, metrics=['mae'])
                
                history = model.fit(
                    X_train_dl, y_train,
                    validation_data=(X_test_dl, y_test),
                    epochs=auto_epochs,
                    batch_size=32,
                    verbose=0
                )
                
                val_loss = history.history['val_loss'][-1]
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_config = {
                        'Layers': num_layers,
                        'Dropout': dropout_rate,
                        'LR': lr
                    }
                    # Extract parameters for PKL compatibility
                    payload = {
                        'architecture_json': model.to_json(),
                        'weights': model.get_weights(),
                        'model_type': 'keras_custom',
                        'input_features_count': X_train.shape[1]
                    }
                    
                    buffer = io.BytesIO()
                    pickle.dump(payload, buffer)
                    best_payload = buffer.getvalue()
                    
                    best_metric_display.success(f"🔥 New Best Validation Loss ({auto_loss.upper()}): **{best_val_loss:.4f}** (Layers: {num_layers}, LR: {lr})")
            
            auto_progress.progress(1.0)
            auto_status.success("AutoML Tuning Complete!")
            st.session_state.trained_model_bytes = best_payload


# -------------------------------------------------------------
# PKL EXPORT (Shared for both tabs)
# -------------------------------------------------------------
st.markdown("---")
if st.session_state.trained_model_bytes is not None:
    st.markdown("### 💾 Export Your Trained Network")
    st.download_button(
        label="Download Best Trained Keras Model Payload (.pkl)",
        data=st.session_state.trained_model_bytes,
        file_name="trained_deep_network.pkl",
        mime="application/octet-stream",
        type="primary"
    )
    st.info("This .pkl payload physically maps the JSON architecture and structural Numpy arrays securely capturing the best parameters from either the Manual Builder or AutoML Tuner.")
