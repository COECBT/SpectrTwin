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
        watch_folder = st.text_input("Folder to Watch", value=st.session_state.watch_folder)
    with f_col2:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        if st.button("📁 Browse", use_container_width=True):
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
                
    # Keep session state synced if user manually types in the field
    if watch_folder != st.session_state.watch_folder:
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

async def start_ws_server(port, message_queue):
    import functools
    async with websockets.serve(functools.partial(ws_handler, message_queue=message_queue), "0.0.0.0", port):
        await asyncio.Future()  # run forever

def run_asyncio_loop(port, message_queue):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    st.session_state.loop = loop
    try:
        loop.run_until_complete(start_ws_server(port, message_queue))
    except Exception as e:
        print(f"Asyncio Loop Error: {e}")

col_run, col_stop = st.columns(2)

with col_run:
    if st.button("▶️ Start Server", use_container_width=True):
        if st.session_state.observer is None:
            if not os.path.exists(watch_folder):
                st.error(f"Folder '{watch_folder}' does not exist. Please create it first.")
            else:
                loaded_model = pickle.load(model_file) if model_file else None
                loaded_pipe = pickle.load(pipeline_file) if pipeline_file else None
                
                # 1. Start WebSockets in background thread
                ws_t = threading.Thread(target=run_asyncio_loop, args=(port, mq), daemon=True)
                ws_t.start()
                st.session_state.ws_thread = ws_t
                
                # 2. Start Watchdog
                event_handler = SpectraFileHandler(mq, loaded_model, loaded_pipe)
                obs = Observer()
                obs.schedule(event_handler, watch_folder, recursive=False)
                obs.start()
                st.session_state.observer = obs
                
                st.rerun()

with col_stop:
    if st.button("⏹️ Stop Server", use_container_width=True):
        if st.session_state.observer is not None:
            st.session_state.observer.stop()
            st.session_state.observer.join()
            st.session_state.observer = None
            
            if st.session_state.loop is not None:
                st.session_state.loop.call_soon_threadsafe(st.session_state.loop.stop)
            
            st.rerun()

if st.session_state.observer is not None:
    st.success(f"🟢 **Server Running!** Monitoring `{watch_folder}` for new `.csv/.txt` spectra.")
    st.info(f"Clients can connect via `ws://YOUR_SERVER_IP:{port}`")
else:
    st.warning("🔴 Server is currently stopped.")
