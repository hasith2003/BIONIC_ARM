import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from config import LABELED_DIR, WINDOW_SIZE, STEP_SIZE
from features import extract_paper_features

def load_and_extract_features(labeled_dir: Path, window_size=250, step_size=50):
    all_files = sorted(labeled_dir.glob("*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {labeled_dir}")
        
    train_files = all_files[:32]
    val_files   = all_files[32:36]
    test_files  = all_files[36:40]
    
    def process_files(file_list):
        X_list, y_list = [], []
        
        for file_path in file_list:
            print(f"   -> Extracting Features: {file_path.name} ...")
            df = pd.read_csv(file_path, header=None)
            
            raw_emg = df.iloc[:, 0:4].values.astype(np.float32)
            labels_column = df.iloc[:, 4].values
            data = np.hstack((raw_emg, labels_column.reshape(-1, 1)))
            
            num_windows = (len(data) - window_size) // step_size + 1
            
            for i in range(num_windows):
                start = i * step_size
                end   = start + window_size
                
                if data[start, 4] != data[end - 1, 4]:
                    continue
                
                window_data = data[start:end, 0:4]
                window_features = []
                
                for ch in range(4):
                    ch_features = extract_paper_features(window_data[:, ch], fs=1000)
                    window_features.extend(ch_features)
                
                X_list.append(window_features)
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

def get_scaled_features():
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_extract_features(LABELED_DIR, WINDOW_SIZE, STEP_SIZE)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test
