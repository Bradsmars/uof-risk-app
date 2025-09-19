from zenml.steps import step
from src.preprocessing import replace_global_na, add_highrisk_label, add_subject_vulnerable, add_tactic_count, add_impact_factor_sum, add_any_firearm, add_multi_location, add_crowd_weapon_interaction, add_na_count_not_stated
import pandas as pd

# i would like cache to be enabled for this step
@step(enable_cache=False)
def feature_eng(df):
    df = replace_global_na(df)
    df = add_highrisk_label(df)
    df = add_subject_vulnerable(df)
    df = add_tactic_count(df)
    df = add_impact_factor_sum(df)
    df = add_any_firearm(df)
    df = add_multi_location(df)
    df = add_crowd_weapon_interaction(df)
    df = add_na_count_not_stated(df)
    return df

