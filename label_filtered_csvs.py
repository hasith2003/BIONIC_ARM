from pathlib import Path
import csv


GESTURE_CLASSES = [
    "Rest",
    "Wrist Extension",
    "Wrist Flexion",
    "Ulnar Deviation",
    "Radial Deviation",
    "Grip",
    "Finger Abduction",
    "Finger Adduction",
    "Supination",
    "Pronation",
]

CLASSES_PER_CYCLE = len(GESTURE_CLASSES)
CYCLES_PER_FILE = 5


def label_for_row(row_index: int, rows_per_class: int) -> tuple[int, str]:
    class_index = (row_index // rows_per_class) % CLASSES_PER_CYCLE
    return class_index, GESTURE_CLASSES[class_index]


import pandas as pd

def label_csv_file(input_path: Path, output_path: Path, rows_per_class: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Read the whole file into RAM instantly as a numpy-backed DataFrame
    df = pd.read_csv(input_path, header=None)
    total_rows = len(df)
    
    # 2. Replicate your mathematical array logic rapidly across the whole column
    # This creates a flat sequence [0,0...1,1...9,9...] repeated 5 times
    class_indices = [(row_idx // rows_per_class) % CLASSES_PER_CYCLE for row_idx in range(total_rows)]
    class_names = [GESTURE_CLASSES[idx] for idx in class_indices]
    
    # 3. Inject the columns directly
    df[4] = class_indices
    df[5] = class_names
    
    # 4. Stream it back out to storage in block chunks
    df.to_csv(output_path, header=False, index=False)

def main() -> None:
    base_dir = Path(__file__).resolve().parent
    filtered_dir = base_dir / "filtered" / "csv"
    output_dir = base_dir / "filtered" / "csv_labeled"

    csv_files = sorted(filtered_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {filtered_dir}")

    with csv_files[0].open("r", encoding="utf-8-sig") as first_file:
        first_file_row_count = sum(1 for _ in first_file)

    if first_file_row_count % (CLASSES_PER_CYCLE * CYCLES_PER_FILE) != 0:
        raise ValueError(
            f"{csv_files[0].name} has {first_file_row_count} rows, which does not divide evenly into "
            f"{CLASSES_PER_CYCLE * CYCLES_PER_FILE} gesture blocks."
        )

    rows_per_class = first_file_row_count // (CLASSES_PER_CYCLE * CYCLES_PER_FILE)

    for input_path in csv_files:
        output_path = output_dir / input_path.name
        label_csv_file(input_path, output_path, rows_per_class)
        print(f"Wrote {output_path}")

    print(
        f"Done. Each row was labeled using {rows_per_class} rows per gesture, "
        f"across {CLASSES_PER_CYCLE} classes and {CYCLES_PER_FILE} cycles."
    )


if __name__ == "__main__":
    main()