
import json
from typing import List
import numpy as np
import pandas as pd
import dice_ml
import logging
import json
from pathlib import Path






#  Thresholded predictions
# -------------------------------------------------------
class TauWrapper:
    """
    wraps a sklearn-style pipeline and applies a custom probability cutoff (tau).
    The tau value is set from metadata (meta['tau']) during model finalization.
    """
    def __init__(self, model, tau):
        self.model = model
        self.tau = tau

    def predict(self, X):
        p = self.model.predict_proba(X)[:, 1]
        return (p >= self.tau).astype(int)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
    


#  Bridge to map DiCE's reduced row back to model schema
# -------------------------------------------------------
class DiceToModelBridge:
    """transforms the smaller DiCE row back to the full model schema (all raw cols).
    
    stores a training template row (self.template) that has all raw columns in the right order.
    in transform: make a DataFrame from X; for every column in the template, if it’s missing in X, insert the template’s value; finally reindex to the template’s order.
    done shape, order, and dtypes now match training exactly.
    
    """
    def __init__(self, template_row: pd.Series):
        # template_row must contain the full model input columns in the right order
        self.template = template_row

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        for column in self.template.index:
            if column not in df.columns:
                df[column] = self.template[column]
        return df[self.template.index]



class WrappedForDice:
    """
    wraps TauWrapper so DiCE predictions always go through the bridge first
    """
    def __init__(self, tau_wrapper, bridge: DiceToModelBridge):
        self.wrapped = tau_wrapper
        self.bridge = bridge
    def predict_proba(self, X):
        X_full = self.bridge.transform(X)
        return self.wrapped.predict_proba(X_full)
    def predict(self, X):
        X_full = self.bridge.transform(X)
        return self.wrapped.predict(X_full)






#  Feature group helpers
# -----------------------


def infer_binary_cols(df: pd.DataFrame) -> List[str]:
    """
    The Goal of this function
    ----
    it finds the columns that behave like simple on and off flags.

    except for "missing" values, a column counts as binary if
      - every value is 'yes' or 'no'     or
      - every value is the number 0 or 1 (int or float)

    returns a list of column names that match those rules.
    """
    assert isinstance(df, pd.DataFrame), " making sure right here that dataframe must be a pandas DataFrame"

    def is_yes_no(values) -> bool:
        """ This function is a tiny checker that decides if a column is a yes/no flag """

        # loop through values
        for value in values:
            # if value not in yes or no then
            if value not in ("yes", "no"):
                # if we hit a non-yes/no value, then fail
                return False
        # if we only saw yes/no values, succeed
        return len(values) > 0  # at least one non-missing value
    

    def is_zero_one(values) -> bool:
        """
        this function is a tiny checker that decides if a column is a 0/1 flag
        it returns true if every value is 0 or 1.
        accepts ints or floats; if any value can't be read as a number, return False.
        """
        
        # If there are no values left after dropping missing values, we can't call it binary
        if len(values) == 0:
            return False
        
        # checking to see if each value is numeric and equals 0 or 1
        for value in values:
            try:
                float_value = float(value)
            except Exception:
                return False # non-numeric like "yes" or "no" fails this numeric rule
            if float_value not in (0.0, 1.0):
                return False # any number other than 0 or 1 breaks the rule
        
        # All values passed the checks - it's a 0/1 column
        return True

    # Start with an empty result list
    binary_columns = []

    # for each column in the df
    for column in df.columns:
        non_missing = df[column].dropna().tolist() # drops the missing values so they don't affect the decision
        if not non_missing:  # if the column is now empty, skip it
            continue
        
        # if the column is all 'yes'/'no' or all 0/1 then add the column name to the result
        if is_yes_no(non_missing) or is_zero_one(non_missing):
            binary_columns.append(column)

    return binary_columns






# Base lists ive been using
CAT_BASE = [
    "person_perceived_age", "person_perceived_gender",
    "person_perceived_ethnicity", "person_mental_health_cond",
    "ced_highest_use"
]
NUM_BASE = [
    "tactic_count", "impact_factor_sum", "na_count_not_stated",
    "subject_vulnerable", "any_firearm", "multi_location", "crowd_weapon"
]



def impute_for_dice(df: pd.DataFrame, binary_cols, cat_cols) -> pd.DataFrame:
    """
    --------------------------
    The goal of this function 
    --------------------------
    making a copy of the dataframe that has no missing values so DiCE won’t crash.

    rules i use (kept super simple):
      - numeric columns        --->  fill missing with 0
      - binary columns (names) --->  fill missing with "no"
      - categorical (names)    --->  fill missing with "missing"
      - any other non-numeric  --->  fill missing with "missing"

    notes
    -------------------------------------------------------------
    - i don’t change dtypes on purpose. i only fill gaps.
    - if i pass a column name that doesn’t exist, i ignore it.
    -------------------------------------------------------------
    """
    # guard: making sure we got a dataframe
    assert isinstance(df, pd.DataFrame), "df must be a pandas DataFrame"


    df_clean = df.copy()

    # 1) looking through the dataframe and grabs the names of columns that are numeric
    numerical_columns = df_clean.select_dtypes(include="number").columns
    if len(numerical_columns) > 0:
        df_clean[numerical_columns] = df_clean[numerical_columns].fillna(0)

    # turn inputs into sets so lookups are easy
    binary_set = set(binary_cols or [])
    categorical_set = set(cat_cols or [])

    # 2) declared binary ---> "no"  (only if the column actually exists)
    for column in binary_set:
        if column in df_clean.columns:
            df_clean[column] = df_clean[column].fillna("no")

    # 3) declared categoricals ---> "missing"
    for column in categorical_set:
        if column in df_clean.columns:
            df_clean[column] = df_clean[column].fillna("missing")

    # 4) everything else that is non-numeric ---> "missing"
    #    (this catches free-text/object columns not in my lists)
    other_obj_cols = [
        column
        for column in df_clean.select_dtypes(exclude="number").columns
        if (column not in binary_set) and (column not in categorical_set)
    ]

    if other_obj_cols:
        df_clean[other_obj_cols] = df_clean[other_obj_cols].fillna("missing")

    return df_clean


# ===============================
#  Schema contract from metadata
# ===============================
def get_expected_raw(pipe, meta: dict) -> List[str]:
    """
    Goal
    -----------
    i want the exact raw input column order the model expects.

    simple rules (in order):
      1) if meta has 'raw_columns' --> use that (best)
    """
    # 1) prefer what training saved
    raw_columns = meta.get("raw_columns")
    # if raw columns exist, use them
    if raw_columns:
        return [str(column)
                for column in raw_columns]










# https://github.com/roshnrf/Diverse-Counterfactual-Explanations-DICE/blob/main/notebook/dice1.py ---> got ideas from here
# https://colab.research.google.com/drive/1nUTTTfcCuxsnZmaJpfvLsxRB4FFaORVK?usp=sharing ---> got ideas from this

def compute_counterfactuals(pipe, meta, total_cfs, dice_train_df, query_df, include_original = True):
    """
    -----------------------
    what this function does
    -----------------------

    it builds counterfactual rows for ONE input (the query) using DiCE.
    the aim: show small, realistic changes that flip the model from HighRisk to Minor
    at the projects threshold (tau).

    how i think about it (simple steps)
    -----------------------------------
    1) figure out model settings (tau) and the exact raw column order it expects
    
    2) decide which columns are immutable vs editable (only UI fields can change)
    
    3) clean up the training frame + 1-row query (simple impute: nums -> 0, binary -> "no",
        cats -> "missing") and reorder to match the model so DiCE wont see nans
        
    4) give DiCE a view with only editable columns, but keep a bridge so predictions
        still hit the FULL model schema at the same tau.
        
    5) set guard rails what features DiCE may vary + safe ranges: any_firearm=0,
        multi_location=0, tactic_count (only goes down), impact_factor_sum (only goes down) 

    6) ask DiCE for counterfactuals with desired_class=0 so results flip toward Minor
    
    7)  print: include the original row (optional) and the (changes-only) CF rows
    
    """
# ------------------------------------------------ 1) Setup: τ + expected raw schema ---------------------------------------------------
    assert "highrisk" in dice_train_df.columns, "dice_train_df must include highrisk"
    decision_threshold_tau = float(meta["tau"]) 
    thresholded_model_wrapper = TauWrapper(pipe,  tau=decision_threshold_tau)  # <- wrapper that enforces tau in predict(), also gets tau from metadata
    expected_raw_column_order_for_model = get_expected_raw(pipe, meta)  # <- exact column order the model saw in training


# -------------------------------------------------- 2) Policy: immutables + simple type groups ------------------------------------------
    immutable_feature_names = [
        "police_force", "year",
        "person_perceived_ethnicity", "person_perceived_gender", "person_perceived_age",
        "person_mental_health_cond", "person_physical_health_cond",
        "subject_vulnerable", "na_count_not_stated",
    ]
    #build simple groups from training columns (present ones only)
    training_feature_names = [column 
                              for column in dice_train_df.columns # looping 
                              if column != "highrisk"]  # <- exclude target variable

    binary_feature_names = [column
                            for column in infer_binary_cols(dice_train_df[training_feature_names]) # looks for yes/no or 0/1
                            if column not in NUM_BASE]
    
    categorical_feature_names = [column
                                 for column in CAT_BASE if
                                 column in training_feature_names]
    
    numeric_feature_names = [column
                             for column in NUM_BASE
                             if column in training_feature_names]
    


# ----------------------------------------------------------- 3) Clean training + build full-schema 1-row query ------------------------------------------------------------

    safe_training_dataframe = impute_for_dice(dice_train_df.copy(), binary_feature_names, categorical_feature_names)  # <- fill NAs so DiCE doesn't break
    
    training_template_row = safe_training_dataframe.drop(columns=["highrisk"], errors="ignore").iloc[0]  # <- taking data types from a real row, sneaky but works
    full_schema_query_row = training_template_row.to_frame().T  # <- transpose to get a 1/single full -row DataFrame
    for column in query_df.columns:  # <- looping through query_df columns
        if column in full_schema_query_row.columns:  # <- check if column exists
            full_schema_query_row.loc[full_schema_query_row.index[0], column] = query_df.iloc[0][column] # <- overwrite with user inputs
            
    # ensure every expected column exists + correct order
    # making sure that every column the model expects is present + in the right order
    for column in expected_raw_column_order_for_model:
        if column not in full_schema_query_row.columns:
            full_schema_query_row[column] = np.nan  # <- if it's missing, we add it as NaN so the order matches perfect
    full_schema_query_row = full_schema_query_row[expected_raw_column_order_for_model]  # <- lock the order to exactly what model expects
    


#------------------------------------------ 4) DiCE view (drop immutables) + schema bridge for faithful scoring ---------------------------------------------------------------

    editable_query_view = full_schema_query_row.drop(columns=immutable_feature_names, errors="ignore").copy()  # <- DiCE can only poke editable cols
    full_schema_prediction_bridge = DiceToModelBridge(template_row=full_schema_query_row.iloc[0])  # <- bridge re-expands reduced rows back to full schema (kinda magic)
    wrapped_model_for_dice = WrappedForDice(thresholded_model_wrapper, full_schema_prediction_bridge)  # <- makes sure every predict() goes thru the bridge

    # DiCE expects a target column
    editable_query_view["highrisk"] = int(wrapped_model_for_dice.predict(editable_query_view.copy())[0])  # <- label the query row so DiCE knows the current class
    
    # dataset DiCE will see = (train minus immutables) + our query
    dataset_view_for_dice = pd.concat(
        [safe_training_dataframe.drop(columns=immutable_feature_names, errors="ignore"), editable_query_view],
        ignore_index=True,
    )  # <- we stitch the query onto a safe training snapshot so DiCE has a proper df to explore

    # one imputation pass for the exact frame DiCE will see
    binary_features_in_dice_view = [column
                                    for column in binary_feature_names
                                    if column in dataset_view_for_dice.columns]
    
    categorical_features_in_dice_view = [column
                                         for column in categorical_feature_names
                                         if column in dataset_view_for_dice.columns]
    
    dataset_view_for_dice = impute_for_dice(dataset_view_for_dice, binary_features_in_dice_view, categorical_features_in_dice_view)  # <- zero NaNs, otherwise DiCE goes weird

    # 1-row query for DiCE (features only; impute once)
    dice_query_features_only = editable_query_view.drop(columns=["highrisk"], errors="ignore").copy()  # <- DiCE wants just features, so we exclude target feature
    dice_query_features_only = impute_for_dice(dice_query_features_only, binary_features_in_dice_view, categorical_features_in_dice_view)  # <- keep it clean, no nulls
    
    

    # -------------------------------------------------------- 5) Wire DiCE (simple) + allow-list of editable UI fields ----------------------------------------------------------

    features_excluding_target = dataset_view_for_dice.drop(columns=["highrisk"], errors="ignore")  # <- drop target feature
    continuous_feature_list = features_excluding_target.select_dtypes(include=["number", "bool"]).columns.tolist()  # <- DiCE needs this split taking continuous features
    categorical_feature_list = [column
                                for column in features_excluding_target.columns
                                if column not in continuous_feature_list]

    dice_data_interface = dice_ml.Data(
        dataframe=dataset_view_for_dice,
        continuous_features=continuous_feature_list,
        categorical_features=categorical_feature_list,
        outcome_name="highrisk",
    )  # <- tells DiCE which columns are numeric vs categorical (it needs this info)
    dice_model_interface = dice_ml.Model(model=wrapped_model_for_dice, backend="sklearn")
    dice_explainer_engine = dice_ml.Dice(dice_data_interface, dice_model_interface, method = "random")  # <- "random" works, "genetic" can be touchy

    allowed_form_fields = {
        #  UI fields
        "ced_highest_use",
        "tactic_count","impact_factor_sum",
        "any_firearm","multi_location","crowd_weapon",
        # optional tactics
        "baton_drawn","baton_used","ced","spit_guard","shield","unarmed_skills",
        "ground_restraint","aep_drawn","aep_used",
        "grouped_irritant_drawn","grouped_irritant_used",
        "dog_deployed","dog_bite",
        "firearms_aimed","firearms_fired",
        "irritant_drawn","irritant_used",
    }
    features_to_vary = sorted([column
                               for column in features_excluding_target.columns
                               if column in allowed_form_fields])  # <- only let DiCE change what users can actually edit

    present_feature_names_set = set(features_excluding_target.columns)  # <- the exact features DiCE sees (no highrisk)

    # setting up restrictions
    # For these features, permitted_range = [0, 0] ⇒ DiCE can only propose value 0 (hard clamp).
    # This enforces a reduce-to-zero policy and steers CFs toward Minor (desired_class=0), but doesn’t guarantee it alone.

    restricted_feature_value_ranges = {}
    if "any_firearm" in present_feature_names_set:
        restricted_feature_value_ranges["any_firearm"] = [0, 0]  # force to 0
    if "multi_location" in present_feature_names_set:
        restricted_feature_value_ranges["multi_location"] = [0, 0]  # force to 0
    if "tactic_count" in present_feature_names_set:
        restricted_feature_value_ranges["tactic_count"] = [0, 0]  # can only reduce (we lock upper to 0 for now, super strict)
    if "impact_factor_sum" in present_feature_names_set:
        restricted_feature_value_ranges["impact_factor_sum"] = [0, 0]  # can only reduce (same idea, keep it safe and boring)
        
        

    # --------------------------------------------------- 6) Generating CFs (we pass ranges + always flip to Minor) --------------------------------------------------------------
    
    
    generated_cf_result = dice_explainer_engine.generate_counterfactuals(
        query_instances=dice_query_features_only,
        total_CFs=int(total_cfs),  # <- how many CFs to try to make
        desired_class=0,  # HighRisk -> Minor at the wrapped τ (we always target the safer class)
        features_to_vary=features_to_vary,
        permitted_range=restricted_feature_value_ranges or None,  # <- bounds so DiCE doesn't suggest wild stuff
        verbose=False,
    )

    # pull changes-only table
    cf_changes_only_table = None
    if hasattr(generated_cf_result, "cf_examples_list") and generated_cf_result.cf_examples_list:
        cf_changes_only_table = generated_cf_result.cf_examples_list[0].final_cfs_df_sparse  # <- sparse view = only diffs from original, super tidy
    if cf_changes_only_table is None and hasattr(generated_cf_result, "visualize_as_dataframe"):
        cf_changes_only_table = generated_cf_result.visualize_as_dataframe(show_only_changes=True)  # <- backup path, just in case
    if cf_changes_only_table is None:
        return pd.DataFrame([{"_note": "no counterfactuals produced."}])  # <- if DiCE gives up, we at least say so

    # dropping immutables from presentation
    cf_changes_only_table = cf_changes_only_table.drop(columns=immutable_feature_names, errors="ignore")  # <- we never changed these, so no point showing them
    

    
    # ----------------------------------------------------------------- 7) Building the  output  ----------------------------------------------------------------------
    
    # if include_original is not there, just return the CFs
    if not include_original:
        output_df = cf_changes_only_table.copy() # making a copy so i dont mess up the original table by accident
        output_df.insert(0, "_row", [f"cf{i+1}"
                                     for i in range(len(output_df))])  # <- # slap a label at the front like CF1, CF2… so its easy to read
        return output_df.reset_index(drop=True) # reset the index so it looks neat

    # take the original query row but drop stuff we never let DiCE change immutables in this case
    original_row = full_schema_query_row.drop(columns=immutable_feature_names, errors="ignore").copy()
    ## fill any missing bits the same way as earlier so taNs ypes are stable (dice hates Nlol)
    original_row = impute_for_dice(original_row, binary_features_in_dice_view, categorical_features_in_dice_view)  # <- keep data type/NA rules consistent
    ## puttig a tag so the first row literally says “original”
    original_row.insert(0, "_row", "original")

    ## now making the CF table that’s parallel to the original (same idea: add labels CF1, CF2…)
    labeled_counterfactual_rows = cf_changes_only_table.copy() # again copy, just being safe
    labeled_counterfactual_rows.insert(0, "_row", [f"cf{i+1}"
                                                   for i in range(len(labeled_counterfactual_rows))]) # label each cf row so I can folow

    # align union (so columns line up even if CFs don't touch some)
    all_aligned_columns = list(dict.fromkeys(original_row.columns.tolist() + labeled_counterfactual_rows.columns.tolist()))  # <- lil trick to keep order stable
    
    # reindex both tables to the same column list so they stack perfectly (no weird mis-align)
    original_row = original_row.reindex(columns=all_aligned_columns)
    labeled_counterfactual_rows = labeled_counterfactual_rows.reindex(columns=all_aligned_columns)

    ## finally stack them: original on top, then all the CFs underneath
    combined_output_table = pd.concat([original_row, labeled_counterfactual_rows], ignore_index=True)  # <- stack original + CFs in one neat table
    
    ## and return a fresh index again, just so it looks tidy in the UI
    return combined_output_table.reset_index(drop=True)


