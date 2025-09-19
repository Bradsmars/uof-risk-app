from zenml.steps import step
from src.combiner import combine_years
import pandas as pd

@step(enable_cache=False)
def merge_data(cleaned_frames):
    return combine_years(cleaned_frames)

