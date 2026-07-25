from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter
from config import LABELED_DIR, PROCESSED_DIR, WINDOW_SIZE, STEP_SIZE, VAL_SIZE, TEST_SIZE, RANDOM_STATE

def load_single_subject_split(
    labeled_dir: Path,
    target_file_idx=0,
    window_size=500,
    step_size=100,
    val_size=0.10,
    test_size=0.10,
    random_state=42,
):
    all_files = sorted(labeled_dir.glob("*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {labeled_dir}")
    
    if target_file_idx >= len(all_files):
        raise ValueError(f"Index {target_file_idx} out of range. Only found {len(all_files)} files.")

    file_path = all_files[target_file_idx]
    print(f"🎯 Target Subject Selected: {file_path.name} (Index: {target_file_idx})")

    X_all, y_all = [], []

    df = pd.read_csv(file_path, header=None)
    data = df.values.copy()

    for ch in range(4):
        ch_signal = data[:, ch].astype(np.float64)
        data[:, ch] = (ch_signal - ch_signal.mean()) / (ch_signal.std() + 1e-8)

    num_windows = (len(data) - window_size) // step_size + 1
    for i in range(num_windows):
        start = i * step_size
        end   = start + window_size

        window_labels = data[start:end, 4]
        if len(np.unique(window_labels)) != 1:
            continue

        X_all.append(data[start:end, 0:4].astype(np.float32))
        y_all.append(int(data[start, 4]))

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.int64)

    if X_all.size == 0:
        raise ValueError(f"No pure windows could be extracted from {file_path.name}.")

    if y_all.min() == 1:
        y_all -= 1

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(X_all))
    X_all = X_all[indices]
    y_all = y_all[indices]

    n_total = len(X_all)
    n_test  = int(n_total * test_size)
    n_val   = int(n_total * val_size)
    n_train = n_total - n_val - n_test

    X_train = X_all[:n_train]
    y_train = y_all[:n_train]
    X_val   = X_all[n_train:n_train + n_val]
    y_val   = y_all[n_train:n_train + n_val]
    X_test  = X_all[n_train + n_val:]
    y_test  = y_all[n_train + n_val:]

    print(f"\n--- Single Subject Split Summary ({file_path.name}) ---")
    print(f"X_train shape : {X_train.shape}")
    print(f"X_val shape   : {X_val.shape}")
    print(f"X_test shape  : {X_test.shape}")
    print(f"Train gesture dist : {dict(sorted(Counter(y_train.tolist()).items()))}")

    return X_train, y_train, X_val, y_val, X_test, y_test, file_path.stem

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test, subject_name = load_single_subject_split(
        LABELED_DIR,
        target_file_idx=0,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print("\n--- Saving Isolated Subject Data to Disk ---")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save_path = PROCESSED_DIR / f"single_{subject_name}_data.npz"

    np.savez_compressed(
        save_path,
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
    )

    print(f"✅ Isolated data successfully packed and saved to: {save_path}")
    print(f"💾 File size on disk: {save_path.stat().st_size / (1024 * 1024):.2f} MB")
