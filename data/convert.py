import os
from pathlib import Path
import pandas as pd


def convert_csvs_to_parquets():
    """
    Converts all csv data files to parquet files within data/raw
    Parquet files are saved in data/processed
    Overall file structure is maintained
    """
    script_dir = Path(__file__).resolve().parent
    raw_dir = script_dir / "raw"
    processed_dir = script_dir / "processed"

    csv_files = list(raw_dir.rglob("*.csv"))

    for csv_path in csv_files:
        try:
            print(f"Processing: {csv_path.relative_to(raw_dir)}")

            relative_path = csv_path.relative_to(raw_dir)

            parquet_path = processed_dir / relative_path.with_suffix(".parquet")

            parquet_path.parent.mkdir(parents=True, exist_ok=True)

            df = pd.read_csv(csv_path)
            df.to_parquet(
                parquet_path, engine="pyarrow", compression="snappy", index=False
            )

        except Exception as e:
            print(f"Error processing {csv_path.name}: {e}")
