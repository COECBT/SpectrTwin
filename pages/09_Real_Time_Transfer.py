import streamlit as st
import pandas as pd
import numpy as np
import pickle
import threading
import asyncio
import websockets
import json
import time
import os
import queue
import socket
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

st.set_page_config(page_title="Real-Time Transfer", layout="wide")
st.title("📡 Real-Time LAN Transfer & Prediction")

try:
    from chatbot import render_chatbot
    render_chatbot("09_Real_Time_Transfer")
except ImportError:
    pass

st.markdown("""
Monitor a local folder for incoming spectra files (e.g., from an active spectrometer). 
The app will automatically preprocess the data, predict using a loaded model, and broadcast the results over the LAN via WebSockets to connected client PCs.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Setup Configuration")
    
    if 'watch_folder' not in st.session_state:
        st.session_state.watch_folder = r"C:\Spectra_Incoming"
        
    f_col1, f_col2 = st.columns([3, 1])
    with f_col1:
        watch_folder = st.text_input("Folder to Watch", value=st.session_state.watch_folder, placeholder="e.g., C:\\Spectra_Incoming or N:\\Data\\Spectra")
    with f_col2:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        if st.button("📁 Browse", use_container_width=True):
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.wm_attributes("-topmost", True)
                root.withdraw()
                chosen_folder = filedialog.askdirectory(parent=root, title="Select folder to watch")
                root.destroy()
                if chosen_folder:
                    st.session_state.watch_folder = chosen_folder.replace("/", "\\")
                    st.rerun()
            except Exception as e:
                st.warning("📌 File browser unavailable on this system (headless environment). Please type the folder path directly.")
                
    # Keep session state synced if user manually types in the field
    if watch_folder and watch_folder != st.session_state.watch_folder:
        st.session_state.watch_folder = watch_folder
        
    port = st.number_input("WebSocket Broadcast Port", min_value=1024, max_value=65535, value=8765)
    
with col2:
    st.subheader("2. Loading Models")
    model_file = st.file_uploader("Upload Trained Model (.pkl)", type=["pkl"])
    pipeline_file = st.file_uploader("Upload Preprocessing Pipeline (.pkl) [Optional]", type=["pkl"])

st.write("---")

# Global message queue for thread-safe cross-thread communication
if "rt_queue" not in st.session_state:
    st.session_state.rt_queue = queue.Queue()
if "observer" not in st.session_state:
    st.session_state.observer = None
if "ws_thread" not in st.session_state:
    st.session_state.ws_thread = None
if "loop" not in st.session_state:
    st.session_state.loop = None
if "server_start_time" not in st.session_state:
    st.session_state.server_start_time = None
if "ws_error" not in st.session_state:
    st.session_state.ws_error = None

mq = st.session_state.rt_queue

class SpectraFileHandler(FileSystemEventHandler):
    def __init__(self, mq, model_obj, pipe_obj):
        self.mq = mq
        self.model = model_obj
        self.pipe = pipe_obj
    
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.csv', '.txt', '.xlsx', '.spa')):
            time.sleep(0.5) # Wait for file to fully write
            try:
                # Load Spectra Data based on extension
                ext = event.src_path.lower().split('.')[-1]
                if ext == 'csv':
                    df = pd.read_csv(event.src_path, header=None)
                elif ext == 'xlsx':
                    df = pd.read_excel(event.src_path, header=None)
                elif ext == 'txt':
                    # Common generic separator for exported text spectra
                    df = pd.read_csv(event.src_path, sep=None, engine='python', header=None)
                else:
                    # For .spa or similar proprietary binary spectra, just create a dummy frame.
                    # Usually requires specific libraries like spectrochempy.
                    df = pd.DataFrame()
                    
                raw_data = df.values.flatten().tolist() if not df.empty else []
                
                prediction = "Model not loaded / Unsupported file"
                if self.model is not None and not df.empty:
                    # Apply processing if available
                    processed = self.pipe.transform(df) if self.pipe else df
                    pred = self.model.predict(processed)[0]
                    prediction = float(pred) if isinstance(pred, (int, float, np.number)) else str(pred)

                payload = {
                    "filename": os.path.basename(event.src_path),
                    "spectra": raw_data,
                    "prediction": prediction,
                    "timestamp": time.time()
                }
                self.mq.put(payload)
                print(f"Broadcasted: {payload['filename']}")
            except Exception as e:
                print(f"Error processing {event.src_path}: {e}")

async def ws_handler(websocket, message_queue):
    while True:
        try:
            if not message_queue.empty():
                msg = message_queue.get_nowait()
                await websocket.send(json.dumps(msg))
            await asyncio.sleep(0.05)
        except websockets.exceptions.ConnectionClosed:
            break
        except Exception as e:
            print(f"WS Error: {e}")
            await asyncio.sleep(1)

def check_port_available(port):
    """Check if a port is available"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('0.0.0.0', port))
            return True
    except OSError:
        return False

async def start_ws_server(port, message_queue):
    import functools
    try:
        async with websockets.serve(functools.partial(ws_handler, message_queue=message_queue), "0.0.0.0", port):
            print(f"✅ WebSocket server started on port {port}")
            await asyncio.Future()  # run forever
    except OSError as e:
        print(f"❌ Port {port} is already in use or inaccessible: {e}")
        raise
    except Exception as e:
        print(f"❌ WebSocket server error: {e}")
        raise

def run_asyncio_loop(port, message_queue):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    st.session_state.loop = loop
    st.session_state.ws_error = None
    try:
        loop.run_until_complete(start_ws_server(port, message_queue))
    except Exception as e:
        error_msg = str(e)
        st.session_state.ws_error = error_msg
        print(f"❌ Asyncio Loop Error: {error_msg}")

col_run, col_stop = st.columns(2)

with col_run:
    if st.button("▶️ Start Server", use_container_width=True):
        # Use session state folder path
        folder_to_monitor = st.session_state.watch_folder
        
        # 1. Check if port is available
        if not check_port_available(port):
            st.error(f"❌ **Port {port} is already in use!**")
            st.info(f"Either:\n- Change the port number above and try again\n- Stop the other server using port {port}")
            st.stop()
        
        # 2. Validate and create folder if needed
        if not os.path.exists(folder_to_monitor):
            try:
                os.makedirs(folder_to_monitor, exist_ok=True)
                st.success(f"✅ Created folder: {folder_to_monitor}")
            except Exception as e:
                st.error(f"❌ Failed to create folder '{folder_to_monitor}': {str(e)}")
                st.stop()
        
        if not os.path.isdir(folder_to_monitor):
            st.error(f"❌ Path '{folder_to_monitor}' is not a directory.")
            st.stop()
        
        if st.session_state.observer is not None:
            st.warning("⚠️ Server is already running. Stop it first before restarting.")
            st.stop()
        
        try:
            # Load models
            loaded_model = None
            loaded_pipe = None
            
            if model_file:
                try:
                    loaded_model = pickle.load(model_file)
                except Exception as e:
                    st.error(f"Failed to load model: {e}")
            
            if pipeline_file:
                try:
                    loaded_pipe = pickle.load(pipeline_file)
                except Exception as e:
                    st.error(f"Failed to load pipeline: {e}")
            
            # Show starting status
            with st.spinner(f"🚀 Starting server on port {port}..."):
                # 1. Start WebSockets in background thread
                ws_t = threading.Thread(target=run_asyncio_loop, args=(port, mq), daemon=True)
                ws_t.start()
                st.session_state.ws_thread = ws_t
                time.sleep(1)  # Give thread time to start and report errors
                
                # Check if WebSocket server started successfully
                if st.session_state.ws_error:
                    st.error(f"❌ **Failed to start WebSocket server**\n{st.session_state.ws_error}")
                    st.info("Try changing the port number or restart the app.")
                    st.session_state.ws_error = None
                    st.stop()
                
                # 2. Start Watchdog
                event_handler = SpectraFileHandler(mq, loaded_model, loaded_pipe)
                obs = Observer()
                obs.schedule(event_handler, folder_to_monitor, recursive=False)
                obs.start()
                st.session_state.observer = obs
                st.session_state.server_start_time = time.time()
            
            st.success("✅ Server started successfully!")
            st.balloons()  # Fun visual feedback
            st.rerun()
            
        except Exception as e:
            st.error(f"Failed to start server: {str(e)}")
            st.stop()

with col_stop:
    if st.button("⏹️ Stop Server", use_container_width=True):
        if st.session_state.observer is not None:
            try:
                st.session_state.observer.stop()
                st.session_state.observer.join(timeout=5)
                st.session_state.observer = None
                
                if st.session_state.loop is not None:
                    st.session_state.loop.call_soon_threadsafe(st.session_state.loop.stop)
                    st.session_state.loop = None
                
                st.success("✅ Server stopped successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error stopping server: {str(e)}")
        else:
            st.info("ℹ️ Server is not running.")

# Display status
st.write("---")
st.subheader("📊 Server Status & Connection Details")

if st.session_state.observer is not None:
    # Get local IP for easy access
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = "127.0.0.1"
    
    # Calculate uptime
    uptime_seconds = int(time.time() - st.session_state.server_start_time) if st.session_state.server_start_time else 0
    uptime_minutes = uptime_seconds // 60
    uptime_text = f"{uptime_minutes}m {uptime_seconds % 60}s" if uptime_minutes > 0 else f"{uptime_seconds}s"
    
    st.success("🟢 **SERVER IS RUNNING**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📍 Server Configuration")
        st.text(f"📂 Monitoring: {st.session_state.watch_folder}")
        st.text(f"🔌 WebSocket Port: {port}")
        st.text(f"⏱️ Uptime: {uptime_text}")
    
    with col2:
        st.markdown("### 🔌 Client Connection URLs")
        st.code(f"ws://127.0.0.1:{port}", language="text")
        st.caption("☝️ Use this for local machine")
        st.code(f"ws://{local_ip}:{port}", language="text")
        st.caption("☝️ Use this for remote client on same network")
    
    st.info(f"""
    ✅ **How to Connect with realtime_client.py:**
    1. Copy the connection URL from above (ws://127.0.0.1:{port} for local or ws://{local_ip}:{port} for network)
    2. Run: `python realtime_client.py`
    3. Paste the URL in the input field
    4. Click "Connect"
    5. Drop spectra files (.csv, .txt, .xlsx, .spa) in: `{st.session_state.watch_folder}`
    6. Watch real-time predictions in the client!
    """)
    
    # Add a simple test button
    col_test1, col_test2 = st.columns(2)
    with col_test1:
        if st.button("📋 Copy Local URL", use_container_width=True):
            st.code(f"ws://127.0.0.1:{port}", language="text")
            st.success("✅ Local URL displayed above - copy it!")
    
    with col_test2:
        if st.button("📋 Copy Network URL", use_container_width=True):
            st.code(f"ws://{local_ip}:{port}", language="text")
            st.success("✅ Network URL displayed above - copy it!")
    
else:
    st.warning("🔴 **SERVER IS STOPPED**")
    st.error("""
    ❌ **Server is not running!**
    
    To start the server:
    1. Enter a folder path above (or accept the default)
    2. (Optional) Upload a trained model and preprocessing pipeline
    3. Click "▶️ Start Server"
    
    Once running, you can connect with `realtime_client.py`
    """)
    
    st.markdown("### ℹ️ Default Settings")
    st.text(f"Default Port: 8765")
    st.text(f"Default Folder: C:\\Spectra_Incoming")
    st.caption("Modify these settings above before starting the server")
