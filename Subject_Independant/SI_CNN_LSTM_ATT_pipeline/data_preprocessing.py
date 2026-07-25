from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from config import LABELED_DIR, WINDOW_SIZE, STEP_SIZE

def load_raw_subject_split(labeled_dir: Path, window_size=500, step_size=100):
    all_files = sorted(labeled_dir.glob("*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {labeled_dir}")
        
    train_files = all_files[:32]
    val_files   = all_files[32:36]
    test_files  = all_files[36:40]
    
    def process_files(file_list):
        X_list, y_list = [], []
        
        for file_path in file_list:
            print(f"   -> Slicing & Normalizing: {file_path.name} ...")
            df = pd.read_csv(file_path, header=None)
            data = df.values.copy()
            
            for ch in range(4):
                ch_signal = data[:, ch].astype(np.float64)
                data[:, ch] = (ch_signal - ch_signal.mean()) / (ch_signal.std() + 1e-8)

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
