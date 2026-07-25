from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from config import LABELED_DIR, WINDOW_SIZE, STEP_SIZE

def load_global_normalize_split(labeled_dir: Path, window_size=500, step_size=100):
    all_files = sorted(labeled_dir.glob("*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {labeled_dir}")
        
    train_files = all_files[:32]
    val_files   = all_files[32:36]
    test_files  = all_files[36:40]
    
    # Pre-scan train files to compute global mean and std for normalization
    print("Scanning train files to compute global mean and std...")
    all_train_data = []
    for file_path in train_files:
        df = pd.read_csv(file_path, header=None)
        all_train_data.append(df.iloc[:, 0:4].values.astype(np.float64))
    
    concat_train = np.vstack(all_train_data)
    global_mean = concat_train.mean(axis=0)
    global_std = concat_train.std(axis=0) + 1e-8
    print(f"Global Train Mean: {global_mean} | Std: {global_std}")

    def process_files(file_list):
        X_list, y_list = [], []
        
        for file_path in file_list:
            print(f"   -> Slicing & Globally Normalizing: {file_path.name} ...")
            df = pd.read_csv(file_path, header=None)
            raw_emg = df.iloc[:, 0:4].values.astype(np.float32)
            labels_column = df.iloc[:, 4].values
            
            # Global normalization
            raw_emg_norm = (raw_emg - global_mean) / global_std
            data = np.hstack((raw_emg_norm, labels_column.reshape(-1, 1)))

            num_windows = (len(data) - window_size) // step_size + 1
            
            for i in range(num_windows):
                start = i * step_size
                end   = start + window_size
                
                if data[start, 4] != data[end - 1, 4]:
                    continue
                
                X_list.append(data[start:end, 0:4].astype(np.float32))
                y_list.append(int(data[start, 4]))
                
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)

    print("--- Processing TRAINING Data ---")
    X_train, y_train = process_files(train_files)
    
    print("\n--- Processing VALIDATION Data ---")
    X_val, y_val = process_files(val_files)
    
    print("\n--- Processing TESTING Data ---")
    X_test, y_test = process_files(test_files)
    
    if y_train.min() == 1:
        y_train -= 1
        y_val   -= 1
        y_test  -= 1

    return X_train, y_train, X_val, y_val, X_test, y_test
