
from pathlib import Path
import pandas as pd
from zenml.steps import step
from src.eda import run_eda_simpler



@step(enable_cache=False)
def eda_step(input_dataframe, output_directory = "eda_simple_plots"):
    """
    running EDA and saving basic plots to output_directory.

    
    returns: absolute path to the folder containing the saved plots.
    """
    # here im making sure the output directory exists (run_eda_simpler also does this,
   
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    saved_path = run_eda_simpler(input_dataframe, output_directory)
    print(f"eda plots saved to: {saved_path}")
    return saved_path
