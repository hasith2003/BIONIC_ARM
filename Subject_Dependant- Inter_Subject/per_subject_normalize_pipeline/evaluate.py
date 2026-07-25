import torch
from config import MODEL_SAVE_PATH, NUM_CLASSES
from dataset import get_dataloaders
from model import CNNBiLSTMAttention

def evaluate_model():
    _, _, test_loader, _ = get_dataloaders()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = CNNBiLSTMAttention(input_dim=4, hidden_dim=64, num_classes=NUM_CLASSES).to(device)

    print(f"Loading weights from: {MODEL_SAVE_PATH}")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()

    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            total_test += batch_y.size(0)
            correct_test += (predicted == batch_y).sum().item()

    final_test_acc = (correct_test / total_test) * 100
    print("====================================================")
    print(f"FINAL PER-SUBJECT NORMALIZED INTER-SUBJECT ACCURACY: {final_test_acc:.2f}%")
    print("====================================================")

if __name__ == "__main__":
    evaluate_model()
