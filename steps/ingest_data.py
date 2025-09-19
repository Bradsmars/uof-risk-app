
#zenml step: load raw excel workbooks for each year and give back
#a dict like {"2022": df, "2023": df, "2024": df}

#no schema cleanups here. im just reading files and a tiny format fix for 2024.

from pathlib import Path
import pandas as pd
from zenml.steps import step

from src.data_ingestion import load_workbook

# config for what to read
# year -> (filename, {sheet_name_in_xlsx: year_value_to_tag}, [other_sheet_names])
WORKBOOKS = {
    "2022": (
        "police-use-of-force-statistics-england-and-wales-open-data-table-2022.xlsx",
        {"2020_21": "2020/21", "2021_22": "2021/22"},  # i will stack these two
        [],
    ),
    "2023": (
        "police-use-of-force-statistics-england-and-wales-open-data-table-2023.xlsx",
        {},
        ["data"],  # single sheet called "data"
    ),
    "2024": (
        "police-use-of-force-statistics-england-and-wales-open-data-table-2024.xlsx",
        {},
        ["data"],  # single sheet called "data"
    ),
}


@step(enable_cache=False)
def ingest_data(data_dir):
    """
    input  : data_dir (folder where the excel files live)
    output : {"2022": df, "2023": df, "2024": df}
    """
    base = Path(data_dir).expanduser()
    frames: dict[str, pd.DataFrame] = {}

    # tiny helpers right here so the flow reads easy
    def merge_2022_sheets(dfs: list[pd.DataFrame]) -> pd.DataFrame:
        # I expect 2 frames (2020_21 and 2021_22) -> stack them
        return pd.concat(dfs, ignore_index=True)
    

    def fix_2024_year_format(df: pd.DataFrame) -> pd.DataFrame:
        # 2024 file has year like "2024_25" -> I want "2024/25"
        if "year" in df.columns:
            df = df.copy()
            df["year"] = df["year"].astype(str).str.replace("_", "/", regex=False)
        return df

    for year, (filename, year_sheets, other_sheets) in WORKBOOKS.items():
        # read raw sheets for this workbook (no cleaning here)
        raw_sheets = load_workbook(base / filename, year_sheets, other_sheets)

        # shape into a single df per year 
        if year == "2022":
            df_year = merge_2022_sheets(raw_sheets)
        else:
            # 2023/2024 give back a single sheet 
            if not raw_sheets:
                raise RuntimeError(f"{year}: no sheets were read from {filename}")
            df_year = raw_sheets[0]

        # tiny fix only for 2024
        if year == "2024":
            df_year = fix_2024_year_format(df_year)

        frames[year] = df_year

    return frames
