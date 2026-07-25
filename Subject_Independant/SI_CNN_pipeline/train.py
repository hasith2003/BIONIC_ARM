import torch
import torch.nn as nn
from tqdm import tqdm
from config import NUM_CLASSES, EPOCHS, MODEL_SAVE_PATH
from dataset import get_dataloaders
from model import EMGSpectrogramCNN

def train_model():
    train_loader, val_loader, _, _ = get_dataloaders()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # STFT parameters: f_bins=33, t_bins=12
    model = EMGSpectrogramCNN(num_classes=NUM_CLASSES, f_bins=33, t_bins=12).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [Train]")
        for batch_X, batch_y in train_bar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += batch_y.size(0)
            correct_train += (predicted == batch_y).sum().item()

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = (correct_train / total_train) * 100

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += batch_y.size(0)
                correct_val += (predicted == batch_y).sum().item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = (correct_val / total_val) * 100

        print(f"Summary -> Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.2f}%")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("--> Saved optimal model weights!")
        print("-" * 80)

if __name__ == "__main__":
    train_model()
