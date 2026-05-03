import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class HPLCDataProcessor:
    
    def __init__(self):
        self.results = {}
        self.peak_ranges = {
            'acid': (8.0, 8.738),
            'main': (8.738, 9.153),
            'base1': (9.153, 9.509),
            'base2': (9.850, 10.315),
            'total': (8.0, 10.315)
        }
    
    @staticmethod
    def detect_separator(file_path):
        separators = [';', ',', '\t', '|']
        
        for sep in separators:
            try:
                df_test = pd.read_csv(file_path, header=None, sep=sep, nrows=5)
                if len(df_test.columns) >= 2:
                    return sep
            except Exception:
                continue
        
        return ','
    
    @staticmethod
    def trapz_integration(input_file, xi, xf):
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"CSV file not found: {input_file}")
        
        try:
            df = pd.read_csv(input_file, sep=None, engine='python')
        except Exception as e:
            raise ValueError(f"Could not read CSV file automatically: {str(e)}")

        
        if len(df.columns) > 5:
            time_col_idx = 4
            signal_col_idx = 5
        elif len(df.columns) >= 2:
            time_col_idx = len(df.columns) - 2
            signal_col_idx = len(df.columns) - 1
        else:
            raise ValueError(f"CSV file has insufficient columns: {len(df.columns)}")
        
        first_row_is_text = False
        try:
            pd.to_numeric(df.iloc[0, time_col_idx])
            pd.to_numeric(df.iloc[0, signal_col_idx])
        except (ValueError, TypeError, IndexError):
            first_row_is_text = True
        
        if first_row_is_text:
            sep = HPLCDataProcessor.detect_separator(input_file)
            df = pd.read_csv(input_file, header=0, sep=sep)
            if len(df.columns) > 5:
                time_col = df.columns[4]
                signal_col = df.columns[5]
            else:
                time_col = df.columns[-2]
                signal_col = df.columns[-1]
        else:
            time_col = time_col_idx
            signal_col = signal_col_idx
        
        df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
        df[signal_col] = pd.to_numeric(df[signal_col], errors='coerce')
        df = df.dropna(subset=[time_col, signal_col])
        
        if len(df) == 0:
            raise ValueError("No valid numeric data found")
        
        mask = (df[time_col] >= xi) & (df[time_col] <= xf)
        filtered_df = df[mask]
        
        if filtered_df.empty:
            raise ValueError(f"No data found between retention times {xi} and {xf}")
        
        x_ax = filtered_df[time_col].tolist()
        y_ax = filtered_df[signal_col].tolist()
        
        if len(x_ax) < 2:
            raise ValueError(f"Insufficient data points ({len(x_ax)}) between {xi} and {xf}")
        
        h = (x_ax[-1] - x_ax[0]) / (len(x_ax) - 1)
        s = y_ax[0] + y_ax[-1]
        for i in range(1, len(y_ax) - 1):
            s = s + 2 * y_ax[i]
        
        area = (h / 2) * s
        return area
    
    def process_hplc_file(self, input_filename, custom_ranges=None):
        try:
            ranges = custom_ranges if custom_ranges else self.peak_ranges
            
            acid = self.trapz_integration(input_filename, *ranges['acid'])
            main = self.trapz_integration(input_filename, *ranges['main'])
            base1 = self.trapz_integration(input_filename, *ranges['base1'])
            base2 = self.trapz_integration(input_filename, *ranges['base2'])
            total = self.trapz_integration(input_filename, *ranges['total'])
            
            acid_pct = (acid / total) * 100
            main_pct = (main / total) * 100
            base1_pct = (base1 / total) * 100
            base2_pct = (base2 / total) * 100
            
            self.results = {
                'success': True,
                'filename': os.path.basename(input_filename),
                'peak_areas': {
                    'acid': acid,
                    'main': main,
                    'base1': base1,
                    'base2': base2,
                    'total': total
                },
                'peak_percentages': {
                    'acid_pct': acid_pct,
                    'main_pct': main_pct,
                    'base1_pct': base1_pct,
                    'base2_pct': base2_pct
                },
                'peak_fractions': {
                    'per_a': acid / total,
                    'per_m': main / total,
                    'per_b1': base1 / total,
                    'per_b2': base2 / total
                },
                'ranges_used': ranges
            }
            
            return self.results
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def load_and_preview_data(self, input_filename, max_rows=1000):
        try:
            file_extension = input_filename.split('.')[-1].lower()
            
            if file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(input_filename, nrows=max_rows)
            elif file_extension == 'txt':
                sep = self.detect_separator(input_filename)
                df = pd.read_csv(input_filename, sep=sep, nrows=max_rows)
            else:
                sep = self.detect_separator(input_filename)
                df = pd.read_csv(input_filename, sep=None, engine='python', nrows=max_rows)

            
            return df
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")


class OptimizationCalculator:
    
    @staticmethod
    def calculate_timing_parameters(t1, t2):
        baseline = 4290
        total_time = 6660
        
        nonpooled_time_1 = t1 - baseline
        pooling_time = t2 - t1
        nonpooled_time_2 = total_time - (t2 - baseline)
        
        nonpooled_time_1_mins = nonpooled_time_1 / 60
        pooling_time_mins = pooling_time / 60
        nonpooled_time_2_mins = nonpooled_time_2 / 60
        
        return {
            'nonpooled_time_1': nonpooled_time_1,
            'pooling_time': pooling_time,
            'nonpooled_time_2': nonpooled_time_2,
            'nonpooled_time_1_mins': nonpooled_time_1_mins,
            'pooling_time_mins': pooling_time_mins,
            'nonpooled_time_2_mins': nonpooled_time_2_mins,
            'baseline': baseline,
            'total_time': total_time,
            't1': t1,
            't2': t2
        }
    
    @staticmethod
    def run_optimization_model(peak_fractions, output_dir='tmp/simulation_files', csv_filename=''):
        try:
            from model_col1 import run_optimization
            
            results = run_optimization(
                per_a=peak_fractions['per_a'],
                per_m=peak_fractions['per_m'],
                per_b1=peak_fractions['per_b1'],
                per_b2=peak_fractions['per_b2'],
                output_dir=output_dir,
                csv_filename=csv_filename
            )
            
            return results
            
        except ImportError:
            return {
                'success': False,
                'error': 'Optimization module (model_col1) not available',
                'error_type': 'ImportError'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }


def plot_chromatogram(df, time_col_idx, signal_col_idx, peak_ranges=None, title="HPLC Chromatogram"):
    fig, ax = plt.subplots(figsize=(14, 6))
    
    if isinstance(time_col_idx, int):
        time_col = df.columns[time_col_idx]
        signal_col = df.columns[signal_col_idx]
    else:
        time_col = time_col_idx
        signal_col = signal_col_idx
    
    ax.plot(df[time_col], df[signal_col], 'b-', linewidth=1.5, label='Signal')
    
    if peak_ranges:
        colors = {'acid': 'red', 'main': 'green', 'base1': 'orange', 'base2': 'purple'}
        alphas = {'acid': 0.2, 'main': 0.2, 'base1': 0.2, 'base2': 0.2}
        
        for peak_name, (start, end) in peak_ranges.items():
            if peak_name != 'total':
                color = colors.get(peak_name, 'gray')
                alpha = alphas.get(peak_name, 0.2)
                ax.axvspan(start, end, alpha=alpha, color=color, label=f'{peak_name.capitalize()} Region')
    
    ax.set_xlabel('Retention Time (min)', fontsize=12)
    ax.set_ylabel('Signal Intensity', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    return fig