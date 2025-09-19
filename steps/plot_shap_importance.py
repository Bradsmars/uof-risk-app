from pathlib import Path
import joblib
import shap
import scipy.sparse as sp
import matplotlib.pyplot as plt
from zenml.steps import step
from src.modeling import get_leakage_cols
from sklearn.calibration import CalibratedClassifierCV

@step(enable_cache=False)
def plot_shap_importance(fe_df,model_path,holdout_year="2023/24",out_dir="saved_model",sample_size=20000,):
    """
    making a SHAP bar chart showing which features mattered most on the holdout split
    mean shap values for high-risk
    """

    fe_copy = fe_df.copy()

    #  first im filtering the holdout split i want to explain
    holdout_split = fe_copy[fe_copy["year"] == holdout_year].reset_index(drop=True)
    # here im checking to see if the holdout split is empty
    if holdout_split.empty:
        # if so, raise an error
        raise ValueError(f"no rows found for holdout year {holdout_year}")

    #  keeping only the model inputs so im going to drop target/leakage columns
    model_inputs_holdout = holdout_split.drop(columns=get_leakage_cols(holdout_split))

    #  sample some rows so shap is fast enough to run
    # 20000 samples
    n_rows_to_sample = min(sample_size, len(model_inputs_holdout))
    # randomly sampling the rows
    sampled_inputs = model_inputs_holdout.sample(n=n_rows_to_sample, random_state=1)

    #  loading the trained pipeline and pulling out the parts I need
    trained_pipeline   = joblib.load(model_path)
    preprocessor       = trained_pipeline.named_steps["prep"]    # columnTransformer
    feature_selector   = trained_pipeline.named_steps["select"]  # selectFromModel
    lightgbm_classifier = trained_pipeline.named_steps["clf"]    # could be CalibratedClassifierCV

    # here i am unwrapping the calibratedclassifiercV to get the base lightGBM model
    # calibratedclassifierCV just wraps my LightGBM and applies a calibration
    # the fitted underlying model is inside calibrated_classifiers_[0].estimator
    base_model = lightgbm_classifier
    if isinstance(lightgbm_classifier, CalibratedClassifierCV):
        if hasattr(lightgbm_classifier, "calibrated_classifiers_") and lightgbm_classifier.calibrated_classifiers_:
            inner = lightgbm_classifier.calibrated_classifiers_[0] #
            # newer sklearn stores the wrapped estimator here:
            if hasattr(inner, "estimator"):
                base_model = inner.estimator
            #  fallback if a different attr name is present
            elif hasattr(inner, "base_estimator"):
                base_model = inner.base_estimator
            else:
                raise TypeError("error")
                

    #  transforming like the pipeline would encode -> select
    encoded_inputs = preprocessor.transform(sampled_inputs)
    #  applying the feature selector
    selected_mask  = feature_selector.get_support()
    #  keeping only the selected features
    selected_inputs = encoded_inputs[:, selected_mask]

    # names for the selected columns (for plot labels)
    all_feature_names       = preprocessor.get_feature_names_out()
    selected_feature_names  = all_feature_names[selected_mask]

    # SHAP prefers dense arrays for plotting
    if sp.issparse(selected_inputs):
        selected_inputs = selected_inputs.toarray()

    #  producing SHAP values TreeExplainer pairs well with LightGBM
    explainer   = shap.TreeExplainer(base_model)  # <<< use the unwrapped model here
    shap_values = explainer.shap_values(selected_inputs)

    # Lgbm binary can return [negative_class, positive_class] here ImM are plotting the positive class/ High risk in this case as i only care for it
    if isinstance(shap_values, list):
        shap_values_to_plot = (shap_values[1]
                               if len(shap_values) > 1
                               else shap_values[0])
    else:
        shap_values_to_plot = shap_values

    #  saving the bar chart
    artifacts_dir = Path(out_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    shap_bar_path = artifacts_dir / "shap_importance_bar.png"

    plt.figure()
    shap.summary_plot(
        shap_values_to_plot,
        selected_inputs,
        feature_names=selected_feature_names,
        plot_type="bar",
        max_display=13,
        show=False,
    )
    plt.title("SHAP Feature Importance (mean SHAP values for High-Risk class)")
    plt.gcf().set_size_inches(10, 6)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(shap_bar_path, dpi=200)
    plt.close()

    # return the path to the saved figure
    return str(shap_bar_path.resolve())
