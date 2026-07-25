from pathlib import Path

WINDOW_SIZE = 500
STEP_SIZE = 100
BATCH_SIZE = 128
NUM_CLASSES = 10
TEST_SIZE = 0.10
VAL_SIZE = 0.10
RANDOM_STATE = 42
EPOCHS = 80

LABELED_DIR = Path("../filtered/csv_labeled").resolve()
PROCESSED_DIR = Path("../processed_data").resolve()
SUBJECT_FILENAME = "single_10_filtered_data.npz"
MODEL_SAVE_PATH = "../best_sd_bilstm_emg.pth"
