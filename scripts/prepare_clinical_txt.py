"""Convert a clinical spreadsheet into one text file per case."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

SKIP_COLUMNS = ["检查时间", "姓名"]
SEQUENCE_COLUMN_CANDIDATES = ["序号", "数据集序号"]


def setup_logger(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "preparation.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )


def find_sequence_column(frame: pd.DataFrame) -> str:
    for column in SEQUENCE_COLUMN_CANDIDATES:
        if column in frame.columns:
            return column
    raise ValueError(f"No sequence column found. Expected one of: {SEQUENCE_COLUMN_CANDIDATES}")


def encode_categorical_column(series: pd.Series):
    codes, unique_values = pd.factorize(series, sort=True)
    mapping = dict(zip(unique_values, range(len(unique_values))))
    return pd.Series(codes, index=series.index), mapping


def process_sheet(sheet_name: str, frame: pd.DataFrame, output_dir: Path) -> dict:
    sequence_column = find_sequence_column(frame)
    frame = frame.drop(columns=SKIP_COLUMNS, errors="ignore")
    frame = frame.loc[frame[sequence_column].notna()].copy()
    frame = frame.loc[frame[sequence_column].astype(str).str.strip().ne("")]
    if frame.empty:
        logging.info("[%s] skipped because no valid case identifier was found.", sheet_name)
        return {"imputation": {}, "encoding": {}}

    imputation_log = {}
    encoding_log = {}

    for column in frame.columns:
        if column == sequence_column:
            continue
        missing_mask = frame[column].isna()
        if missing_mask.any():
            if is_numeric_dtype(frame[column]):
                fill_value = frame[column].median()
                strategy = "median"
            else:
                fill_value = frame[column].mode(dropna=True).iloc[0]
                strategy = "mode"
            frame.loc[missing_mask, column] = fill_value
            imputation_log[column] = {"strategy": strategy, "fill_value": fill_value}

        if column != sequence_column and not is_numeric_dtype(frame[column]):
            frame[column], mapping = encode_categorical_column(frame[column])
            encoding_log[column] = mapping

    class_dir = output_dir / sheet_name
    class_dir.mkdir(parents=True, exist_ok=True)
    for _, row in frame.iterrows():
        case_id = row[sequence_column]
        if isinstance(case_id, float) and case_id.is_integer():
            case_id_text = f"{int(case_id):02d}"
        else:
            case_id_text = str(case_id)
        values = row.drop(labels=sequence_column).tolist()
        formatted = []
        for value in values:
            if isinstance(value, float) and value.is_integer():
                formatted.append(str(int(value)))
            else:
                formatted.append(str(value))
        (class_dir / f"{sheet_name}_{case_id_text}.txt").write_text(" ".join(formatted) + "\n", encoding="utf-8-sig")

    return {"imputation": imputation_log, "encoding": encoding_log}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a clinical Excel workbook into per-case text files.")
    parser.add_argument("--excel", required=True, help="Path to the source workbook.")
    parser.add_argument("--output", default="data/clinical_txt", help="Directory for generated text files.")
    args = parser.parse_args()

    output_dir = Path(args.output)
    setup_logger(output_dir)
    workbook = pd.ExcelFile(args.excel)
    summary = {}
    for sheet_name in workbook.sheet_names:
        frame = pd.read_excel(workbook, sheet_name=sheet_name)
        summary[sheet_name] = process_sheet(sheet_name, frame, output_dir)

    with open(output_dir / "preparation_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()
