"""
Just small Input and output helpers for the Police Use-of-Force Excel files.

No ZenML stuff. No cleaning. We only read sheets and add a year tag when needed.
"""
from pathlib import Path
import pandas as pd


# --------------------  low-level helpers  ------------
def load_sheet(file_path, sheet_name):
    # make sure file exists
    if not file_path.is_file():
        raise FileNotFoundError(f"file ain't found: {file_path}")

    try:
        return pd.read_excel(file_path, sheet_name=sheet_name)
    except ValueError as error:  # pandas raises ValueError if the sheet is missing
        raise RuntimeError(
            f"sheets '{sheet_name}' not found brad in {file_path.name}"
        ) from error


def add_year_column(df, year_value):
    """
    add a 'year' column at position 0 if it's not there already.
    i did this keep it simple because later code expects a 'year' column.
    """
    if "year" not in df.columns:
        df = df.copy()
        df.insert(0, "year", year_value)
    return df


# ───────────────────────────── public API (easy to read) ─────────────────────────────
def load_workbook(file_path, year_sheets, other_sheets):
    """
    read the sheets we care about from one workbook.

    inputs
       file_path   : path to the .xlsx
       year_sheets : mapping like {"2020_21": "2020/21", "2021_22": "2021/22"}
                      we will read each sheet and insert a 'year' column with the mapped value
      other_sheets: list of sheet names we want as-is (no extra 'year' tagging here)

    output
      - list of dataframes in the order we read them (keep it predictable)
    """
    path = Path(file_path).expanduser()

    frames = [] #this here contains list of pandas dataframes

    #  read sheets that need a 'year' value stamped in
    for sheet_name, year_value in year_sheets.items():
        df = load_sheet(path, sheet_name)
        df = add_year_column(df, year_value)
        frames.append(df)

    #  read any other sheets as-is (no extra columns added)
    for sheet_name in other_sheets:
        df = load_sheet(path, sheet_name)
        frames.append(df)

    return frames
