import torch
from torch.utils.data import TensorDataset, DataLoader
from config import LABELED_DIR, WINDOW_SIZE, STEP_SIZE, BATCH_SIZE
from data_preprocessing import load_raw_subject_split

def get_dataloaders():
    X_train, y_train, X_val, y_val, X_test, y_test = load_raw_subject_split(
        LABELED_DIR, WINDOW_SIZE, STEP_SIZE
    )
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds   = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    test_ds  = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader, y_train
