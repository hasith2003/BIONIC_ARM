from pathlib import Path
import csv


def count_csv_rows(csv_path: Path) -> int:
	with csv_path.open("r", newline="", encoding="utf-8-sig") as file_handle:
		return sum(1 for _ in csv.reader(file_handle))


def main() -> None:
	filtered_dir = Path(__file__).resolve().parent / "filtered" / "csv"
	total_rows = 0

	for csv_path in sorted(filtered_dir.glob("*.csv")):
		row_count = count_csv_rows(csv_path)
		total_rows += row_count
		print(f"{csv_path.name}: {row_count} lines")

	print(f"Total lines across all filtered CSV files: {total_rows}")


if __name__ == "__main__":
	main()
