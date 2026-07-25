from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from scipy import signal
from config import LABELED_DIR, WINDOW_SIZE, STEP_SIZE

def load_stft_subject_split(labeled_dir: Path, window_size=250, step_size=50):
    all_files = sorted(labeled_dir.glob("*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {labeled_dir}")
        
    train_files = all_files[:32]
    val_files   = all_files[32:36]
    test_files  = all_files[36:40]
    
    def process_files_to_stft(file_list):
        X_list, y_list = [], []
        
        for file_path in file_list:
            print(f"   -> Slicing & Computing STFT: {file_path.name} ...")
            df = pd.read_csv(file_path, header=None)
            raw_emg       = df.iloc[:, 0:4].values.astype(np.float32)
            labels_column = df.iloc[:, 4].values
            
            data = np.hstack((raw_emg, labels_column.reshape(-1, 1)))
            num_windows = (len(data) - window_size) // step_size + 1
            
            for i in range(num_windows):
                start = i * step_size
                end   = start + window_size
                
                if data[start, 4] != data[end - 1, 4]:
                    continue
                
                window_data = data[start:end, 0:4]
                channels_spectrograms = []
                
                for ch in range(4):
                    f, t, Sxx = signal.spectrogram(
                        window_data[:, ch],
                        fs=1000, nperseg=64, noverlap=48
                    )
                    log_Sxx = np.log1p(np.abs(Sxx).astype(np.float32))
                    channels_spectrograms.append(log_Sxx)
                
                X_list.append(np.stack(channels_spectrograms, axis=0))
                y_list.append(int(data[start, 4]))
                
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)

    print("--- Extracting TRAINING Spectrograms ---")
    X_train, y_train = process_files_to_stft(train_files)
    
    print("\n--- Extracting VALIDATION Spectrograms ---")
    X_val, y_val = process_files_to_stft(val_files)
    
    print("\n--- Extracting TESTING Spectrograms ---")
    X_test, y_test = process_files_to_stft(test_files)
    
    if y_train.min() == 1:
        y_train -= 1
        y_val   -= 1
        y_test  -= 1

    spec_min = X_train.min(axis=(0, 2, 3), keepdims=True)
    spec_max = X_train.max(axis=(0, 2, 3), keepdims=True)
    denom = spec_max - spec_min
    denom[denom == 0] = 1e-8

    X_train = np.clip((X_train - spec_min) / denom, 0, 1)
    X_val   = np.clip((X_val   - spec_min) / denom, 0, 1)
    X_test  = np.clip((X_test  - spec_min) / denom, 0, 1)

    return X_train, y_train, X_val, y_val, X_test, y_test
