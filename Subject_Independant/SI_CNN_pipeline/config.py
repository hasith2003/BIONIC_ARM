from pathlib import Path

WINDOW_SIZE = 250
STEP_SIZE = 50
BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS = 20
RANDOM_STATE = 42

LABELED_DIR = Path("../filtered/csv_labeled").resolve()
MODEL_SAVE_PATH = "../best_si_cnn_emg.pth"
