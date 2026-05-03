import numpy as np
import pandas as pd
import seaborn as sns
from numpy.fft import rfft, irfft, rfftfreq
import matplotlib.pyplot as plt
from typing import Union, Optional, Tuple

class FFTProcessor:
    
    def __init__(self, X_train, X_test, y_train, y_test, threshold: float = 1e3, sampling_interval: float = 20e-3):     
        self.X_train = X_train  
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.threshold = threshold
        self.sampling_interval = sampling_interval
        self.fitted_threshold_ = None
        self.is_fitted_ = False
        self.n_features_ = None
        
    def _get_values(self, data):
        if hasattr(data, 'values'):
            return data.values
        return data
    
    def _get_columns(self, data):
        if hasattr(data, 'columns'):
            return data.columns
        return [f'feature_{i}' for i in range(data.shape[1] if data.ndim > 1 else len(data))]
    
    def filter_signal(self, signal, threshold: float = None):
        if threshold is None:
            threshold = self.fitted_threshold_
        
        if threshold is None:
            threshold = self.threshold
            
        signal_values = self._get_values(signal)
        
        if signal_values.ndim > 1:
            signal_values = signal_values.flatten()
        
        if len(signal_values) == 0:
            return signal_values
        
        fourier = rfft(signal_values)
        frequencies = rfftfreq(signal_values.size, d=self.sampling_interval)
        fourier[frequencies > threshold] = 0
        filtered = irfft(fourier)
        
        if len(filtered) != len(signal_values):
            if len(filtered) > len(signal_values):
                filtered = filtered[:len(signal_values)]
            else:
                padded = np.zeros(len(signal_values))
                padded[:len(filtered)] = filtered
                filtered = padded
        
        return filtered
    
    def fit(self, X_train=None, start_col: int = 0, end_col: Optional[int] = None):
        if X_train is None:
            X_train = self.X_train
            
        X_values = self._get_values(X_train)
        
        if X_values.ndim != 2:
            raise ValueError(f"Expected 2D input, got {X_values.ndim}D")
        
        self.start_col = max(0, start_col)
        if end_col is None:
            self.end_col = X_values.shape[1]
        else:
            self.end_col = min(end_col, X_values.shape[1])
            
        self.fitted_threshold_ = self.threshold if self.threshold is not None else 1e3
        self.n_features_ = self.end_col - self.start_col
        self.columns_ = self._get_columns(X_train)
        self.is_fitted_ = True
        
        print(f"FFT Processor fitted:")
        print(f"  - Input shape: {X_values.shape}")
        print(f"  - Feature range: columns {self.start_col} to {self.end_col}")
        print(f"  - Number of features to process: {self.n_features_}")
        print(f"  - Threshold: {self.fitted_threshold_}")
        
        return self
    
    def _transform_single(self, X, return_dataframe: bool = False):
        X_values = self._get_values(X)
        
        if X_values.ndim != 2:
            raise ValueError(f"Expected 2D input, got {X_values.ndim}D")
            
        n_samples, n_features = X_values.shape
        
        available_features = min(n_features, self.end_col) - self.start_col
        if available_features <= 0:
            raise ValueError(f"No features available to process. Data has {n_features} features, "
                           f"but processor expects features {self.start_col} to {self.end_col}")
        
        filtered_data = np.zeros_like(X_values)
        
        if self.start_col > 0:
            filtered_data[:, :self.start_col] = X_values[:, :self.start_col]
        if self.end_col < n_features:
            filtered_data[:, self.end_col:] = X_values[:, self.end_col:]
        
        for sample_idx in range(n_samples):
            spectrum = X_values[sample_idx, self.start_col:min(n_features, self.end_col)]
            filtered_spectrum = self.filter_signal(spectrum, threshold=self.fitted_threshold_)
            end_idx = min(n_features, self.end_col)
            filtered_data[sample_idx, self.start_col:end_idx] = filtered_spectrum[:end_idx-self.start_col]
        
        if return_dataframe and hasattr(X, 'columns'):
            return pd.DataFrame(filtered_data, columns=X.columns, index=X.index if hasattr(X, 'index') else None)
        elif return_dataframe:
            columns = [f'feature_{i}' for i in range(filtered_data.shape[1])]
            return pd.DataFrame(filtered_data, columns=columns)
        
        return filtered_data
    
    def transform(self, X=None, return_dataframe: bool = False):
        if not self.is_fitted_:
            raise ValueError("Must call fit() before transform(). Use: processor.fit()")
        
        if X is not None:
            return self._transform_single(X, return_dataframe)
        
        print("Transforming both X_train and X_test:")
        print(f"  - X_train shape: {self._get_values(self.X_train).shape}")
        print(f"  - X_test shape: {self._get_values(self.X_test).shape}")
        
        X_train_filtered = self._transform_single(self.X_train, return_dataframe)
        X_test_filtered = self._transform_single(self.X_test, return_dataframe)
        
        print(f"  - X_train_filtered shape: {X_train_filtered.shape}")
        print(f"  - X_test_filtered shape: {X_test_filtered.shape}")
        
        return X_train_filtered, X_test_filtered
    
    def fit_transform(self, X_train=None, start_col: int = 0, end_col: Optional[int] = None, return_dataframe: bool = False):
        if X_train is None:
            self.fit(self.X_train, start_col, end_col)
            return self.transform(return_dataframe=return_dataframe)
        return self.fit(X_train, start_col, end_col).transform(X_train, return_dataframe)
    
    def save_filtered_data(self, data: np.ndarray, filename: str = 'filtered_data.csv'):
        np.savetxt(filename, data, delimiter=',')
        print(f"Filtered data saved to {filename}")
    
    def load_excel_data(self, filepath: str, sheet_name: str = 'Phe'):
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        print(f"Loaded data: {df.shape} from {filepath}[{sheet_name}]")
        return df
    
    def plot_comparison(self, original_data=None, filtered_data=None, sample_idx: int = 0):
        if original_data is None:
            original_data = self.X_test
        if filtered_data is None:
            _, filtered_data = self.transform()
            
        original_values = self._get_values(original_data)
        filtered_values = self._get_values(filtered_data)
        
        if sample_idx >= original_values.shape[0]:
            sample_idx = 0
            
        plt.figure(figsize=(15, 6))
        
        original_spectrum = original_values[sample_idx, :]
        filtered_spectrum = filtered_values[sample_idx, :]
        
        plt.plot(original_spectrum, 'b-', alpha=0.7, label='Original', linewidth=2)
        plt.plot(filtered_spectrum, 'r-', alpha=0.8, label='FFT Filtered', linewidth=2)
        
        plt.title(f"FFT Filtering Comparison - Sample {sample_idx}", size=16)
        plt.xlabel("Feature Index", size=14)
        plt.ylabel("Intensity", size=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def get_summary(self):
        if not self.is_fitted_:
            return "FFT Processor not fitted yet. Call fit() first."
        
        return {
            'fitted': self.is_fitted_,
            'threshold': self.fitted_threshold_,
            'feature_range': f"{self.start_col} to {self.end_col}",
            'n_features_processed': self.n_features_,
            'sampling_interval': self.sampling_interval
        }
    
    def process_all(self, X=None, start_col: int = 0, end_col: Optional[int] = None, threshold: float = None):
        if X is None:
            if threshold is not None:
                self.threshold = threshold
            return self.fit(self.X_train, start_col, end_col).transform(return_dataframe=False)
        if threshold is not None:
            self.threshold = threshold
        return self.fit(X, start_col, end_col).transform(X, return_dataframe=False)