import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataAugmentor:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def _get_values(self, data):
        if hasattr(data, 'values'):
            return data.values
        else:
            return data
    
    def _get_columns(self, data):
        if hasattr(data, 'columns'):
            return data.columns
        else:
            return [f'feature_{i}' for i in range(data.shape[1])]
    
    def _ensure_1d(self, y_data):
        y_values = self._get_values(y_data)
        if isinstance(y_values, (pd.Series, pd.DataFrame)):
            return y_values.values.flatten()
        else:
            return np.array(y_values).flatten()

    def add_spectra(self, num_copies=None):
        X, y = self.X, self.y
        X_values = self._get_values(X)
        y_values = self._ensure_1d(y)
        columns = self._get_columns(X)

        n = len(X_values)
        num_new_samples = num_copies if num_copies is not None else n
        X_new, y_new = [], []

        for _ in range(num_new_samples):
            i, j = np.random.choice(n, size=2, replace=False)
            X_new.append((X_values[i] + X_values[j]) / 2)
            y_new.append((y_values[i] + y_values[j]) / 2)

        X_aug = pd.DataFrame(np.vstack([X_values, X_new]), columns=columns)
        y_aug = pd.Series(np.concatenate([y_values, y_new]), name='target')
        return X_aug, y_aug

    def mixup(self, num_copies=2, alpha=0.4):
        X, y = self.X, self.y
        X_values = self._get_values(X)
        y_values = self._ensure_1d(y)
        columns = self._get_columns(X)
        
        n = len(X_values)
        X_mix, y_mix = [], []

        for _ in range(num_copies):
            i, j = np.random.choice(n, size=2, replace=False)
            lam = np.random.beta(alpha, alpha)
            X_mix.append(lam * X_values[i] + (1 - lam) * X_values[j])
            y_mix.append(lam * y_values[i] + (1 - lam) * y_values[j])

        X_aug = pd.DataFrame(np.vstack([X_values] + [X_mix]), columns=columns)
        y_aug = pd.Series(np.concatenate([y_values, y_mix]), name='target')
        return X_aug, y_aug

    def spectral_shift(self, num_copies=2, shift_range=3):
        X, y = self.X, self.y
        X_values = self._get_values(X)
        y_values = self._ensure_1d(y)
        columns = self._get_columns(X)
        
        n = len(X_values)
        X_shifted, y_shifted = [], []

        for _ in range(num_copies):
            for i in range(n):
                shift = np.random.randint(-shift_range, shift_range + 1)
                X_shifted.append(np.roll(X_values[i], shift))
                y_shifted.append(y_values[i])

        X_aug = pd.DataFrame(np.vstack([X_values] + [X_shifted]), columns=columns)
        y_aug = pd.Series(np.concatenate([y_values, y_shifted]), name='target')
        return X_aug, y_aug

    def gaussian_noise(self, num_copies=2, mean=0.0, std=0.01, random_state=None):
        X, y = self.X, self.y
        X_values = self._get_values(X)
        y_values = self._ensure_1d(y)
        columns = self._get_columns(X)
        
        rng = np.random.RandomState(random_state)

        X_noisy = [X_values + rng.normal(loc=mean, scale=std, size=X_values.shape) for _ in range(num_copies)]
        y_noisy = [y_values for _ in range(num_copies)]

        X_aug = pd.DataFrame(np.vstack([X_values] + X_noisy), columns=columns)
        y_aug = pd.Series(np.concatenate([y_values] + y_noisy), name='target')
        return X_aug, y_aug