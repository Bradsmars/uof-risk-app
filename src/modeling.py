import numpy as np
import os, random
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV

leakage_cols = (
    ["highrisk", "person_injury_level", "officer_injury_level",
     "person_injured_from_force", "officer_injured_assault",
     "officer_injured", "officer_injured_intentional_assault", 
     "officer_injured_no_assault"])

def get_leakage_cols(df):
    dynamic_cols = [column 
                    for column in df.columns 
                    if column.startswith("outcome_")]
    
    cols = leakage_cols + dynamic_cols
    return [column 
            for column in cols 
            if column in df.columns]


def build_model(X_train):
    # column groups ---------------------------------
    binary_cols = [column 
                   for column in X_train.columns 
                   if X_train[column].isin(["yes", "no"]).all()]

    cat_cols = [
        "person_perceived_age", "person_perceived_gender",
        "person_perceived_ethnicity", "person_mental_health_cond", "ced_highest_use"
    ]

    numeric_cols = [
    "tactic_count", "impact_factor_sum", "na_count_not_stated",
    "subject_vulnerable", "any_firearm", "multi_location", "crowd_weapon"
]
    force_col = ["police_force"]

    bin_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="no", missing_values=pd.NA)),
        ("ohe",    OneHotEncoder(drop="if_binary", dtype=np.int8)),
    ])
    
    
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing", missing_values=pd.NA)),
        ("ohe",    OneHotEncoder(handle_unknown="ignore")),
    ])
    
    
    num_pipe = SimpleImputer(strategy="constant", fill_value=0, missing_values=pd.NA)
    
    
    force_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("te",     TargetEncoder()),
    ])

    preproc = ColumnTransformer([
        ("bin",   bin_pipe,  binary_cols),
        ("cat",   cat_pipe,  cat_cols),
        ("num",   num_pipe,  numeric_cols),
        ("force", force_pipe, force_col), #force is a high-cardinality categorical variable so im target encoding it
    ])
# didnt get enough time to do try other classifiers, but LGBM was a good performer
    selector = SelectFromModel(
        LGBMClassifier(objective="binary", n_estimators=300,
                       max_depth=5, class_weight="balanced", random_state=SEED,threshold="median"
    ))
  
    calibrated_clf = CalibratedClassifierCV(
        estimator =LGBMClassifier(objective="binary", n_estimators=600,
                                    learning_rate=0.05, max_depth=7, random_state=SEED),
        method="isotonic",   # or "sigmoid" (Platt) im going with isotonic because it tends to perform better for large samples
        cv=5                 # K-fold split for calibration within training data
    )

    return ImbPipeline([
        ("prep",   preproc),
        ("select", selector),
        ("clf",    calibrated_clf),
    ])
