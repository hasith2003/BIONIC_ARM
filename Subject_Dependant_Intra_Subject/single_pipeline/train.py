import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from tqdm import tqdm
from config import NUM_CLASSES, EPOCHS, MODEL_SAVE_PATH
from dataset import get_dataloaders
from model import CNNBiLSTMAttention

def check_pred_distribution(model, loader, device, label=""):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch_X, _ in loader:
            preds = model(batch_X.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
    print(f"  [{label}] Pred dist: {dict(sorted(Counter(all_preds).items()))}")

def train_model():
    train_loader, val_loader, _, y_train = get_dataloaders()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = CNNBiLSTMAttention(input_dim=4, hidden_dim=64, num_classes=NUM_CLASSES).to(device)

    counts = np.bincount(y_train)
    weights = 1.0 / counts.astype(np.float32)
    weights = weights / weights.sum() * NUM_CLASSES
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5
    )

    EARLY_STOP_PATIENCE = 10
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [Train]", unit="batch")
        for batch_X, batch_y in train_bar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += batch_y.size(0)
            correct_train += (predicted == batch_y).sum().item()

            train_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{(predicted == batch_y).sum().item() / batch_y.size(0) * 100:.2f}%",
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
            })

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = (correct_train / total_train) * 100

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [Val  ]", unit="batch", leave=False)
        with torch.no_grad():
            for batch_X, batch_y in val_bar:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += batch_y.size(0)
                correct_val += (predicted == batch_y).sum().item()
                val_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = (correct_val / total_val) * 100
        scheduler.step(epoch_val_acc)

        print(f"Summary -> Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if (epoch + 1) % 5 == 0:
            print("  Diagnostics:")
            check_pred_distribution(model, train_loader, device, "Train")
            check_pred_distribution(model, val_loader, device, "Val  ")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"--> ✅ Saved best model! Val Acc: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"    No improvement. Patience: {patience_counter}/{EARLY_STOP_PATIENCE}")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n⛔ Early stopping at epoch {epoch+1}!")
            break

        print("-" * 80)

    print(f"\n✅ Training complete! Best Val Acc: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train_model()
