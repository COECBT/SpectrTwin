# SpectraTwin

A comprehensive spectral analysis and machine learning pipeline application built with Streamlit. SpectraTwin provides tools for experimental design, data visualization, preprocessing, model training, neural network building, and real-time analysis of spectral data.

## Features

- **Experimental Design**: Plan and design spectral experiments with statistical rigor
- **Data Visualization**: Interactive visualization of spectral data and analysis results
- **Data Preprocessing**: Advanced preprocessing techniques for spectral data
- **Model Training**: Train machine learning models (PCA, MPLS, OPLS) on spectral data
- **Neural Network Builder**: Design and train custom neural networks
- **Control Charts**: Statistical process control and quality monitoring
- **Full Pipeline**: End-to-end automated analysis workflow
- **Model Prediction**: Make predictions using trained models
- **One-Click Pipeline**: Streamlined single-step analysis
- **Real-Time Transfer**: Live spectral data transfer and analysis
- **HPLC Integration**: High-Performance Liquid Chromatography data analysis

## Project Structure

```
spectraTwin/
├── Home.py                          # Main Streamlit app
├── pages/                           # Streamlit multi-page app pages
│   ├── 00_Experimental design.py
│   ├── 01_Data_Visualization.py
│   ├── 02_Preprocessing.py
│   ├── 03_Model_Training.py
│   ├── 04_Neural_Network_Builder.py
│   ├── 05_Control chart.py
│   ├── 06_Full_Pipeline.py
│   ├── 07_Model_prediction.py
│   ├── 08_One_Click_Pipeline.py
│   ├── 09_Real_Time_Transfer.py
│   └── 10_HPLC.py
├── Logo/                            # Application logos and assets
├── Core Modules:
│   ├── pca.py                       # Principal Component Analysis
│   ├── mpls.py                      # Modified Partial Least Squares
│   ├── opls.py                      # Orthogonal Partial Least Squares
│   ├── hplc.py                      # HPLC data processing
│   ├── FFT.py                       # Fast Fourier Transform
│   ├── preprocess.py                # Data preprocessing utilities
│   ├── data_augmentation.py         # Data augmentation techniques
│   ├── evaluate_design.py           # Experimental design evaluation
│   ├── midel.py                     # Model utilities
│   └── chatbot.py                   # AI chatbot integration
├── Streamlit Pipelines:
│   ├── streamlit_fast_pipeline.py
│   ├── streamlit_full_pipeline.py
│   └── streamlit_optimized_pipeline.py
├── realtime_client.py               # Real-time data client
└── Api.txt                          # API configuration file
```

## Prerequisites

- Python 3.8+
- pip package manager

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Naveen-567/SpectrTwin.git
cd SpectraTwin
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. **Set up Groq API Key** (IMPORTANT):
   - Visit [Groq Console](https://console.groq.com) and create a free account
   - Generate an API key from your account dashboard
   - Create or update the `Api.txt` file in the project root:
     ```
     GroQ_API_key = your_api_key_here
     ```
   - Replace `your_api_key_here` with your actual Groq API key
   - **⚠️ DO NOT commit this file with your API key to version control**

## Running the Application

### Locally

#### Main Application
```bash
streamlit run Home.py
```

#### Fast Pipeline
```bash
streamlit run streamlit_fast_pipeline.py
```

#### Full Pipeline
```bash
streamlit run streamlit_full_pipeline.py
```

#### Optimized Pipeline
```bash
streamlit run streamlit_optimized_pipeline.py
```

#### Real-Time Data Transfer Client
**Note:** `realtime_client.py` is a **separate standalone application** designed to run on a different system (instrument/data source) for real-time spectral data transfer.

**Setup and Usage:**
1. Install dependencies on the data source system:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure the client:
   - Edit `realtime_client.py` with your:
     - Server IP/hostname (where SpectraTwin app is running)
     - Port configuration
     - Data source parameters

3. Run on the data source system:
   ```bash
   python realtime_client.py
   ```

4. Access real-time data in SpectraTwin via **09_Real_Time_Transfer** page

**Architecture:**
```
Data Source System        →    SpectraTwin System
(realtime_client.py)           (Main App)
  ↓                               ↓
 Instrument/Device          Streamlit Interface
  ↓                               ↓
Network Connection          View Live Data
```

The application will open in your default web browser at `http://localhost:8501`

### Deploy on Streamlit Cloud

**Follow the complete deployment guide:** [DEPLOYMENT.md](DEPLOYMENT.md)

**Quick Summary:**
1. Go to [Streamlit Cloud](https://streamlit.io/cloud) and sign in with GitHub
2. Click "New app" and select this repository
3. Set your Groq API key in app settings → Secrets:
   ```toml
   GROQ_API_KEY = "your_api_key_here"
   ```
4. Deploy!

**Common Issues:**
- If chatbot errors occur, it will gracefully disable (app still works)
- Ensure `GROQ_API_KEY` is set in Streamlit Cloud Secrets
- Check deployment logs if issues persist

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed troubleshooting.

## Usage

1. **Experimental Design**: Start by designing your experiment with proper statistical parameters
2. **Data Preparation**: Preprocess your spectral data using the Preprocessing module
3. **Model Training**: Train appropriate models based on your data characteristics
4. **Analysis**: Run predictions and analyze results using trained models
5. **Quality Control**: Monitor results using control charts

## Configuration

### API Setup
Edit `Api.txt` to configure your Groq API credentials:
```
GroQ_API_key = your_groq_api_key
```

## Dependencies

Key dependencies include:
- streamlit
- numpy
- pandas
- scikit-learn
- groq
- plotly
- scipy

See full list in `requirements.txt`

## Real-Time Features

The application supports real-time spectral data transfer through a client-server architecture:

### Real-Time Client (`realtime_client.py`)
**Important:** This is a **separate standalone application** that runs on your **data source system** (instrument/sensor system), NOT on the SpectraTwin main app system.

**Features:**
- Streams spectral data in real-time to SpectraTwin server
- Runs independently on data acquisition hardware
- Supports continuous data monitoring
- Configurable network endpoints

**Usage:**
1. Run on your data source/instrument system:
   ```bash
   python realtime_client.py
   ```

2. Configure connection parameters:
   - Server IP: where SpectraTwin is running
   - Port: configured in both client and server
   - Data format: depends on your instrument

3. Monitor incoming data in SpectraTwin via **Page 09: Real-Time Transfer**

### Real-Time Processing (`09_Real_Time_Transfer.py`)
- Web interface for viewing live spectral data
- Real-time visualization and analysis
- Data logging and archival
- Alert/notification system

**System Architecture:**
```
┌─────────────────────────────┐
│  Data Source System         │
│  (Separate Computer)        │
│                             │
│  • realtime_client.py       │
│  • FTIR/Raman/NIR Device    │
│  • Sensors/Instruments      │
└────────────────┬────────────┘
                 │ Network Connection
                 │ (TCP/IP or other)
                 ↓
┌─────────────────────────────┐
│  SpectraTwin Main System    │
│  (Streamlit App)            │
│                             │
│  • 09_Real_Time_Transfer.py │
│  • Live Dashboard           │
│  • Data Analysis            │
└─────────────────────────────┘
```

## Troubleshooting

**Real-Time Connection Issues:**
- Verify network connectivity between systems
- Check firewall settings allow data transfer
- Ensure `realtime_client.py` is running on data source system
- Check configured IP and port match in both systems

**Issue: Groq API Key Error**
- Ensure your API key is correctly set in `Api.txt`
- Verify the key is active on the Groq console
- Check your internet connection

**Issue: Missing Dependencies**
- Run `pip install -r requirements.txt` again
- Ensure you're using the correct Python version (3.8+)

**Issue: Port Already in Use**
- The app uses port 8501 by default
- To use a different port: `streamlit run Home.py --server.port 8502`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or support, please open an issue on the GitHub repository.

---

**Note**: This application is designed for spectral data analysis in research and industrial settings. Ensure you have appropriate expertise when interpreting results for critical applications.
