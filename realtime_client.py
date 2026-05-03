import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import asyncio
import websockets
import json
import threading
import queue
import pandas as pd
import os
from datetime import datetime

class RealTimeSpectraClient:
    def __init__(self, root):
        self.root = root
        self.root.title("📡 Real-Time Spectra Client")
        self.root.geometry("900x600")
        
        # State
        self.ws_loop = None
        self.connected = False
        self.data_queue = queue.Queue()
        self.log_data = [] # Store logs if user wants to save
        
        self._build_ui()
        self.root.after(100, self.process_queue)
        
    def _build_ui(self):
        # Top Frame for Controls
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X, side=tk.TOP)
        
        ttk.Label(control_frame, text="Server IP (ws://...):").pack(side=tk.LEFT, padx=5)
        self.ip_entry = ttk.Entry(control_frame, width=30)
        self.ip_entry.insert(0, "ws://127.0.0.1:8765")
        self.ip_entry.pack(side=tk.LEFT, padx=5)
        
        self.connect_btn = ttk.Button(control_frame, text="Connect", command=self.toggle_connection)
        self.connect_btn.pack(side=tk.LEFT, padx=10)
        
        self.save_btn = ttk.Button(control_frame, text="Save Log As...", command=self.save_log)
        self.save_btn.pack(side=tk.RIGHT, padx=5)
        
        self.auto_save_var = tk.BooleanVar(value=False)
        self.auto_save_chk = ttk.Checkbutton(control_frame, text="Auto-Save Live", variable=self.auto_save_var, command=self.toggle_autosave)
        self.auto_save_chk.pack(side=tk.RIGHT, padx=10)
        self.auto_save_file = ""
        
        self.status_lbl = ttk.Label(control_frame, text="Disconnected", foreground="red")
        self.status_lbl.pack(side=tk.RIGHT, padx=15)
        
        # Middle Frame for Plot
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.line, = self.ax.plot([], [], lw=2, color='blue')
        self.ax.set_title("Real-Time Spectrum")
        self.ax.set_xlabel("Wavenumber / Features")
        self.ax.set_ylabel("Intensity")
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Bottom Frame for Predictions
        pred_frame = ttk.Frame(self.root, padding="10")
        pred_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.filename_lbl = ttk.Label(pred_frame, text="Last File: None", font=("Helvetica", 10))
        self.filename_lbl.pack(side=tk.LEFT)
        
        self.pred_lbl = ttk.Label(pred_frame, text="Prediction: N/A", font=("Helvetica", 14, "bold"), foreground="green")
        self.pred_lbl.pack(side=tk.RIGHT)
        
    def toggle_connection(self):
        if not self.connected:
            url = self.ip_entry.get()
            self.ws_thread = threading.Thread(target=self.start_async_loop, args=(url,), daemon=True)
            self.ws_thread.start()
            self.connect_btn.config(text="Disconnecting...")
            self.connect_btn.state(['disabled'])
        else:
            if self.ws_loop:
                self.ws_loop.call_soon_threadsafe(self.ws_loop.stop)
            self.connected = False
            self.status_lbl.config(text="Disconnected", foreground="red")
            self.connect_btn.config(text="Connect")
            
    def start_async_loop(self, url):
        self.ws_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.ws_loop)
        try:
            self.ws_loop.run_until_complete(self.listen_to_server(url))
        except Exception as e:
            print("Loop stopped:", e)
            
    async def listen_to_server(self, url):
        try:
            async with websockets.connect(url) as websocket:
                self.data_queue.put({"type": "status", "msg": "Connected", "color": "green"})
                self.connected = True
                while True:
                    message = await websocket.recv()
                    data = json.loads(message)
                    self.data_queue.put({"type": "data", "payload": data})
        except Exception as e:
            self.data_queue.put({"type": "status", "msg": f"Error: {e}", "color": "red"})
            self.connected = False
            
    def process_queue(self):
        try:
            while not self.data_queue.empty():
                item = self.data_queue.get_nowait()
                if item["type"] == "status":
                    self.status_lbl.config(text=item["msg"], foreground=item["color"])
                    if item["msg"] == "Connected":
                        self.connect_btn.config(text="Disconnect")
                        self.connect_btn.state(['!disabled'])
                    else:
                        self.connect_btn.config(text="Connect")
                        self.connect_btn.state(['!disabled'])
                elif item["type"] == "data":
                    payload = item["payload"]
                    self.update_ui(payload)
        except Exception as e:
            print("Queue error:", e)
        finally:
            self.root.after(50, self.process_queue)
            
    def update_ui(self, payload):
        spectra = payload.get("spectra", [])
        prediction = payload.get("prediction", "N/A")
        filename = payload.get("filename", "Unknown")
        
        # Log it locally
        log_entry = {
            "timestamp": datetime.fromtimestamp(payload.get("timestamp", time.time())).strftime('%Y-%m-%d %H:%M:%S'),
            "filename": filename,
            "prediction": prediction
        }
        self.log_data.append(log_entry)
        
        # Real-time Auto-saving
        if self.auto_save_var.get() and self.auto_save_file:
            try:
                # Append to file simply
                is_csv = self.auto_save_file.endswith(".csv")
                sep = "," if is_csv else "\t"
                # write header if file is new
                write_header = not os.path.exists(self.auto_save_file)
                pd.DataFrame([log_entry]).to_csv(self.auto_save_file, mode='a', header=write_header, index=False, sep=sep)
            except Exception as e:
                print("Autosave error:", e)

        # Update text
        self.filename_lbl.config(text=f"Last File: {filename}")
        self.pred_lbl.config(text=f"Prediction: {prediction}")
        
        # Update Plot
        if spectra:
            self.line.set_xdata(range(len(spectra)))
            self.line.set_ydata(spectra)
            self.ax.relim()
            self.ax.autoscale_view()
            self.canvas.draw()
            
    def toggle_autosave(self):
        if self.auto_save_var.get():
            from tkinter import filedialog
            chosen = filedialog.asksaveasfilename(
                title="Select Auto-Save File (Will append data)",
                defaultextension=".txt",
                filetypes=[("Text File", "*.txt"), ("CSV File", "*.csv")]
            )
            if chosen:
                self.auto_save_file = chosen
            else:
                self.auto_save_var.set(False)
        else:
            self.auto_save_file = ""

    def save_log(self):
        if not self.log_data:
            messagebox.showinfo("No Data", "No predictions received yet to save.")
            return
            
        from tkinter import filedialog
        save_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text File", "*.txt"), ("CSV File", "*.csv")],
            title="Save Prediction Log"
        )
        if not save_path:
            return
            
        df = pd.DataFrame(self.log_data)
        try:
            sep = "," if save_path.endswith(".csv") else "\t"
            df.to_csv(save_path, index=False, sep=sep)
            messagebox.showinfo("Success", f"Log saved to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save log:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = RealTimeSpectraClient(root)
    root.mainloop()