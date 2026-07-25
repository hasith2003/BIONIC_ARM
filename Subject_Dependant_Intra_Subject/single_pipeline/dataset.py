import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from config import PROCESSED_DIR, SUBJECT_FILENAME, BATCH_SIZE

def get_dataloaders():
    load_path = PROCESSED_DIR / SUBJECT_FILENAME
    print(f"Loading isolated subject data from: {load_path} ...")
    
    data = np.load(load_path)
    X_train = data['X_train']
    y_train = data['y_train']
    X_val   = data['X_val']
    y_val   = data['y_val']
    X_test  = data['X_test']
    y_test  = data['y_test']

    print("\n✅ Personal subject data successfully loaded into memory!")
    print("--- Sanity Check ---")
    print(f"X_train shape : {X_train.shape}")
    print(f"X_val shape   : {X_val.shape}")
    print(f"X_test shape  : {X_test.shape}")

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader, y_train
