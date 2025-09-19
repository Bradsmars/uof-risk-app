
import pandas as pd

def combine_years(frames):
    """
    goal
    ----
    here im stacking the yearly dataframes top-to-bottom, but first make sure they all
    share the same columns (full union). missing columns get filled with pd.NA

    input  : {"2022": df, "2023": df, "2024": df}  (i expect a dictionary of dataFrames)
    output : one big dataframe with a stable sorted column order
    """
    # guards (keep errors early and clear)
    assert hasattr(frames, "values"), "frames should be a dict-like of dataFrames"
    dfs = list(frames.values())
    if not dfs:
        return pd.DataFrame()
    assert all(isinstance(df, pd.DataFrame) for df in dfs), "all values must be dataFrames"

    # first make the full set of columns across all years (sorted for a stable order)
    all_cols = sorted({column 
                       for df in dfs
                       for column in df.columns})

    # second align each df to that union (reindex adds any missing cols with pd.NA)
    aligned = [df.reindex(columns=all_cols, fill_value=pd.NA) for df in dfs]

    # third stack them (ignore_index gives a clean 0..N-1 index)
    return pd.concat(aligned, ignore_index=True, sort=False)

