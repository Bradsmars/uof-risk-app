import sys
from pathlib import Path
# making the parent folder importable so i can import from my source folder
sys.path.append(str(Path(__file__).resolve().parents[1]))  # allowing my source imports

import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from src.cf_utils import TauWrapper, compute_counterfactuals  # contains the model wrapper and CF generator

# inspiration for streamlit template was taken from https://streamlit.io/gallery

# basic page setup so it looks decent
st.set_page_config(page_title="High-Risk Prediction and Counterfactuals", layout="wide")
st.title("High-Risk Prediction and Counterfactuals")

#------------------------------------------------------------------------------ Find artifacts in saved_model --------------------------------------------------------------------------------

# im looking for the newest model file like "highrisk_*.pkl"
MODEL_PATHS = sorted(Path("saved_model").glob("highrisk_*.pkl"))
LATEST_MODEL_PATH = MODEL_PATHS[-1]  # last = newest when sorted by name pattern I used
MODEL_METADATA_PATH = LATEST_MODEL_PATH.with_name(LATEST_MODEL_PATH.stem + "_meta.json")
DICE_TRAIN_PARQUET_PATH = LATEST_MODEL_PATH.parent / "dice_train.parquet"  # training frame used for DiCE (has highrisk)




@st.cache_resource
def load_artifacts():
    """
    loading the trained pipeline, metadata json, and the parquet used for DiCE.
    cache it so every rerun isnt slow. if stuff is missing, i tell the user.
    """
    pipeline = joblib.load(LATEST_MODEL_PATH)  # trained sklearn LightGBM pipeline
    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    dice_training_dataframe = pd.read_parquet(DICE_TRAIN_PARQUET_PATH)  # includes highrisk column
    # the exact raw column order the model expects. fall back to whatever dice_train has (minus target)
    expected_raw_columns = metadata.get(
        "raw_columns",
        list(dice_training_dataframe.drop(columns=["highrisk"], errors="ignore").columns),
    )
    return pipeline, metadata, dice_training_dataframe, expected_raw_columns

try:
    pipeline, metadata, dice_train, expected_raw = load_artifacts()
except Exception as error:
    # dont crash ugly, just explain what went wrong
    st.error(f"failed to load artefacts: {error}")
    st.stop()

# grabbing the decision threshold τ the training phase set (or default)
decision_threshold_tau = float(metadata.get("tau", 0.044))
# this wrapper lets me do predict() at a fixed tau + get predict_proba()
thresholded_model_wrapper = TauWrapper(pipeline, tau=decision_threshold_tau)

# cute sidebar info + a manual cache clear button
with st.sidebar:
    st.markdown(f"**Model:** `{LATEST_MODEL_PATH.name}`")
    st.markdown(f"**Threshold τ:** `{decision_threshold_tau:.3f}`")
    if st.button("Clear cache & reload"):
        st.cache_resource.clear()
        st.rerun()


#------------------------------------------------------------------------------- Small helpers -----------------------------------------------------------------------------------
# 
def options_for_column(column_name):
    """
    grab selectbox options from dice_train so I only let users pick values that actually exist.
    'missing' is the neutral token for NA. super useful so DiCE doesn't explode later.
    """
    if column_name in dice_train.columns:
        # filter out the literal string "missing" so I can re-add it at the front
        unique_values = sorted(value 
                               for value in dice_train[column_name].dropna().unique().tolist()
                               if value != "missing")
        # re-adding "missing" as a neutral option
        return ["missing"] + unique_values
    # if the column doesn't exist, return just "missing"
    return ["missing"]



# Only the fields I actually collect in the form (this keeps the UI focused)
FORM_COLUMNS = [
    # core fields
    "person_perceived_age", "person_perceived_gender", "person_perceived_ethnicity",
    "person_mental_health_cond", "ced_highest_use", "police_force", "year",
    "tactic_count", "impact_factor_sum", "na_count_not_stated", "subject_vulnerable",
    "any_firearm", "multi_location", "crowd_weapon",
    # optional tactics
    "baton_drawn", "baton_used", "ced", "spit_guard", "shield", "unarmed_skills",
    "ground_restraint", "aep_drawn", "aep_used",
    "grouped_irritant_drawn", "grouped_irritant_used",
    "dog_deployed", "dog_bite",
    "firearms_aimed", "firearms_fired",
    "irritant_drawn", "irritant_used",
]

def put_value(model_input_dataframe: pd.DataFrame, column_name: str, value, is_numeric: bool = False):
    """
    write a single value into the 1-row input dataframe, converting to int when needed.
    'missing' becomes NaN so the pipeline treats it like proper missing.
    """
    model_input_dataframe.loc[model_input_dataframe.index[0], column_name] = (
        int(value) if is_numeric and value is not None else (value if value != "missing" else np.nan)
    )



#------------------------------------------------------------ UI layout -----------------------------------------------------------------------------------


left_column, right_column = st.columns([1, 1.15])

with left_column:
    st.subheader("Enter incident details")

    with st.form("incident"):
        # Categorical (strings)
        person_perceived_age       = st.selectbox("Age group", options=options_for_column("person_perceived_age"), index=0)
        person_perceived_gender    = st.selectbox("Gender", options=options_for_column("person_perceived_gender"), index=0)
        person_perceived_ethnicity = st.selectbox("Ethnicity", options=options_for_column("person_perceived_ethnicity"), index=0)
        person_mental_health_cond  = st.selectbox("Mental health condition", options=options_for_column("person_mental_health_cond"), index=0)
        ced_highest_use            = st.selectbox("CED highest use", options=options_for_column("ced_highest_use"), index=0)
        police_force               = st.selectbox("Police force", options=options_for_column("police_force"), index=0)
        year                       = st.selectbox("Year", options=options_for_column("year"), index=0)

        # Numeric (plain ints)
        tactic_count        = st.number_input("Tactic count", min_value=0, step=1, value=0)
        impact_factor_sum   = st.number_input("Impact factor sum", min_value=0, step=1, value=0)
        na_count_not_stated = st.number_input("NA count not stated", min_value=0, step=1, value=0)
        subject_vulnerable  = st.number_input("Subject vulnerable", min_value=0, step=1, value=0)
        any_firearm         = st.selectbox("Any firearm", options=[0, 1], index=0)
        multi_location      = st.selectbox("Multi-location", options=[0, 1], index=0)
        crowd_weapon        = st.selectbox("Crowd weapon", options=[0, 1], index=0)

        # optional tactical flags inside an expander 
        with st.expander("Additional tactics (optional)"):
            # Less-lethal / public order stuff
            baton_drawn   = st.selectbox("Baton drawn?", ["missing", "no", "yes"], 0)
            baton_used    = st.selectbox("Baton used?", ["missing", "no", "yes"], 0)
            spit_guard    = st.selectbox("Spit guard applied?", ["missing", "no", "yes"], 0)
            shield        = st.selectbox("Shield deployed?", ["missing", "no", "yes"], 0)
            unarmed_skills = st.selectbox("Unarmed skills used?", ["missing", "no", "yes"], 0)
            ground_restraint = st.selectbox("Ground restraint used?", ["missing", "no", "yes"], 0)
            aep_drawn     = st.selectbox("AEP drawn?", ["missing", "no", "yes"], 0)
            aep_used      = st.selectbox("AEP used?", ["missing", "no", "yes"], 0)
            grouped_irritant_drawn = st.selectbox("Grouped irritant drawn?", ["missing", "no", "yes"], 0)
            grouped_irritant_used  = st.selectbox("Grouped irritant used?", ["missing", "no", "yes"], 0)

            # Canine
            dog_deployed  = st.selectbox("Dog deployed?", ["missing", "no", "yes"], 0)
            dog_bite      = st.selectbox("Dog bite?", ["missing", "no", "yes"], 0)

            # Firearms
            firearms_aimed = st.selectbox("Firearms aimed?", ["missing", "no", "yes"], 0)
            firearms_fired = st.selectbox("Firearms fired?", ["missing", "no", "yes"], 0)

        # the actual submit button (this returns True when I click)
        submitted = st.form_submit_button("Predict")

with right_column:
    st.subheader("Results")

    if submitted:
   
            # 1. building a 1-row input with correct data types
            # -------------------------------------------
            #  copying a training row (minus target) so all data types match exactly what the model saw.
            training_template_row = dice_train.drop(columns=["highrisk"], errors="ignore").iloc[0].copy()
            model_input_dataframe = training_template_row.to_frame().T  # 1-row DataFrame

          
            # 2. Write ALL the form fields into model_input_dataframe
            # ----------------------------------------------------
            # I loop a list of (column_name, value, is_numeric) so its obvious whats getting set.
            for column_name, value, is_numeric in [
                # categoricals
                ("person_perceived_age",       person_perceived_age,       False),
                ("person_perceived_gender",    person_perceived_gender,    False),
                ("person_perceived_ethnicity", person_perceived_ethnicity, False),
                ("person_mental_health_cond",  person_mental_health_cond,  False),
                ("ced_highest_use",            ced_highest_use,            False),
                ("police_force",               police_force,               False),
                ("year",                       year,                       False),

                # numeric / binary-as-int
                ("tactic_count",        tactic_count,        True),
                ("impact_factor_sum",   impact_factor_sum,   True),
                ("na_count_not_stated", na_count_not_stated, True),
                ("subject_vulnerable",  subject_vulnerable,  True),
                ("any_firearm",         any_firearm,         True),
                ("multi_location",      multi_location,      True),
                ("crowd_weapon",        crowd_weapon,        True),

                # optional tactics (strings yes/no/missing)
                ("baton_drawn", baton_drawn, False),
                ("baton_used", baton_used, False),
                ("spit_guard", spit_guard, False),
                ("shield", shield, False),
                ("unarmed_skills", unarmed_skills, False),
                ("ground_restraint", ground_restraint, False),
                ("aep_drawn", aep_drawn, False),
                ("aep_used", aep_used, False),
                ("grouped_irritant_drawn", grouped_irritant_drawn, False),
                ("grouped_irritant_used",  grouped_irritant_used,  False),
                ("dog_deployed",  dog_deployed,  False),
                ("dog_bite",      dog_bite,      False),
                ("firearms_aimed", firearms_aimed, False),
                ("firearms_fired", firearms_fired, False),
            ]:
                # each loop I set exactly one field.
                put_value(model_input_dataframe, column_name, value, is_numeric=is_numeric)

            # mini-derived flag: if the highest CED use looks like "none/missing", set parent 'ced' to "no", else "yes"
            ced_parent_flag = "no" if str(ced_highest_use).strip().lower() in {"missing", "no", "none", "not_stated"} else "yes"
            put_value(model_input_dataframe, "ced", ced_parent_flag)
            
            #this should be done for othe fields come back to this when you have time

      
            # 3. Neutralise every non-form column
            # ----------------------------------------------------
            # any feature not shown in the UI gets set to NaN so the display + DiCE dont get weird surprises.
            non_form_columns = [column 
                                for column in model_input_dataframe.columns
                                if column not in FORM_COLUMNS]
            
            # setting all non-form columns to NaN
            model_input_dataframe.loc[model_input_dataframe.index[0], non_form_columns] = np.nan


            # 4. Reorder to the exact schema the model expects
            # -----------------------------------------------------
            # this is super important so predict_proba lines up column-wise.
            model_input_dataframe = model_input_dataframe.reindex(columns=expected_raw)

   
            # 5) Predict probability + label at decision threshold
            # -----------------------------------------------------
            probability_highrisk = float(thresholded_model_wrapper.predict_proba(model_input_dataframe)[:, 1][0])  # predicting probability of high risk
            label_highrisk = int(probability_highrisk >= decision_threshold_tau)

            st.markdown(
                f" The Probability of the incident being 'HighRisk' is: `{probability_highrisk:.3f}`"
                f" &nbsp;&nbsp; but the tau is &nbsp;&nbsp; τ: `{decision_threshold_tau:.3f}`"
                f" &nbsp;&nbsp;→&nbsp;&nbsp; Prediction: {['Minor','HighRisk'][label_highrisk]}"
            )

         
            # 6 Show a SHAP global importance image if we saved one
            # ------------------------------------------------------
            shap_image_path = next(iter(LATEST_MODEL_PATH.parent.glob("shap*importance*.png")), None) or \
                              next(iter(LATEST_MODEL_PATH.parent.glob("shap*.png")), None)
            if shap_image_path:
                st.subheader("Global feature importance (SHAP)")
                st.image(str(shap_image_path), use_column_width=True)

       
            # 7 Generate counterfactuals (include original row too)
            # ---------------------------------------------------
            st.subheader("Counterfactual suggestions")
            counterfactual_dataframe = compute_counterfactuals(
                pipe=pipeline,
                meta=metadata,
                total_cfs=6,
                dice_train_df=dice_train,
                query_df=model_input_dataframe,
                include_original=True,
            )

            # only showing the fields the that i can actually change + prediction columns
            display_columns = ["_row"] + [
                column 
                for column in FORM_COLUMNS
                if column in counterfactual_dataframe.columns
            ]
            counterfactual_view = counterfactual_dataframe[display_columns]

            st.dataframe(counterfactual_view, use_container_width=True)





        
