import logging
import pandas as pd
from zenml import step


from src.cleaning_data import Cleaner_2020_2023
from src.cleaning_data import Cleaner2024

@step(enable_cache=False)
def clean_data(raw_frames):
    """ returns one cleaned df   per year in a dictionary."""
    return {
        "2022": Cleaner_2020_2023().clean(raw_frames["2022"]),
        "2023": Cleaner_2020_2023().clean(raw_frames["2023"]),
        "2024": Cleaner2024().clean(raw_frames["2024"]),
    }
