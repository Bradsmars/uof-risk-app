from steps.eda_step import eda_step
from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.merge_data import merge_data
from steps.feature_eng import feature_eng
from steps.finalise_model import finalise_model
from steps.plot_shap_importance import plot_shap_importance

# inspiration for pipeline implementation: 
# https://www.youtube.com/watch?v=dPmH3G9NQtY&t=7655s&ab_channel=AyushSingh
# Notebook (Ipynb) was first used for coding in order to get a better understanding of how to carry out this pipeline.
# Notebook consisted of functions which were applicable to this machine learning framework.

# inspiration for turning code into functions from jupyter notebook: https://www.reddit.com/r/MachineLearning/comments/q344pp/notebook_to_production_d/
# Missing implementations due to time constrains (mlflow and other tracking mechanisms within zenml flow)

@pipeline
def train_pipeline(data_path):
    raw_yearly   = ingest_data(data_dir=data_path)
    cleaned_dict = clean_data(raw_yearly)
    merged_df    = merge_data(cleaned_dict)
    fe_df        = feature_eng(merged_df)
    _ = eda_step(input_dataframe=fe_df, output_directory="eda_plots")
                             
    # final retrain and save artifacts (handles filtering internally)
    model_path = finalise_model(fe_df, version="v1", tau=0.044, out_dir="saved_model")
    plot_shap_importance(fe_df, model_path)
