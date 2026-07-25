from pathlib import Path

WINDOW_SIZE = 500
STEP_SIZE = 100
BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS = 80
RANDOM_STATE = 42

LABELED_DIR = Path("../filtered/csv_labeled").resolve()
MODEL_SAVE_PATH = "../best_si_cnn_lstm_att_emg.pth"
