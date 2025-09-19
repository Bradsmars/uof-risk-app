import pandas as pd

# ---------- helper for Replacing columns containing not_stated with Na's as not stated has the same meaning as pd.na ----------

# ---------- to check if code works initially used jupyter notebook, then turned into functions ----------

NA_CODES = {"not_stated", "", None}

def replace_global_na(df):
    return df.replace(NA_CODES, pd.NA)

# ---------- helper for engineering target column ----------

def make_highrisk(row) -> str:
    """return HighRisk if record matches Severe/Moderate rules; else Minor
    - here im making the highrisk column using the injury levels and injury flags
    from whether a person is injured from force or when an officer is injured.
    
    """
    subj_lvl = str(row["person_injury_level"]).lower()
    off_lvl  = str(row["officer_injury_level"]).lower()
    subj_inj = str(row["person_injured_from_force"]).lower()
    off_inj  = str(row["officer_injured"]).lower()

    # Severe
    if subj_lvl in {"death", "severe"} or off_lvl == "severe":
        return "HighRisk"

    # Moderate
    if (
        subj_lvl == "minor"
        or off_lvl == "minor"
        or subj_inj == "yes"
        or off_inj == "yes"
        or row["officer_injured_intentional_assault"] is True
        or row["officer_injured_no_assault"] == "yes"
    ):
        return "HighRisk"

    return "Minor"


def add_highrisk_label(df):
    """ adds a highrisk target column and returns new DF"""
    df = df.copy()
    df["highrisk"] = df.apply(make_highrisk, axis=1)
    return df

# ---------- helper for engineering subject vulnerability column ----------

def add_subject_vulnerable(df):
    """
    flag subjects who are (a) under 18, (b) 65 +, or (c) have a recorded
    mental‑health impact factor.
    produces a binary int8 column subject_vulnerable (binary)
    """
    df = df.copy()

    age_risk = df["person_perceived_age"].isin(
        {"under_11", "11_17", "65_and_over"}
    )

    mental_health_risk = df["person_mental_health_cond"].isin(
        {"yes", "mental_health_condition", "physical_mental_health_condition"}
    )

    df["subject_vulnerable"] = (age_risk | mental_health_risk).astype("int8")
    return df


# ---------- helper for engineering tactic count column ----------

def add_tactic_count(df):
    """
    creating tactic_count = number of tactic columns flagged "yes"
    """
    df = df.copy()

    tactic_cols = [c for
                   c in df.columns
                   if c.startswith(
        ("ced_", "baton_", "dog_", "firearms_", "irritant_",
         "ground_restraint", "shield", "spit_guard",
         "unarmed_skills", "limb_body_restraints")
    )]
    
    df["tactic_count"] = (df[tactic_cols] == "yes").sum(axis=1).astype("int8")
    return df

# ---------- helper for engineering impact factor sum column ----------

def add_impact_factor_sum(df):
    """
    creating impact_factor_sum = number of positive impact‑factor flags
    """
    df = df.copy()

    impact_cols = [c 
                   for c in df.columns
                   if c.startswith("impact_factor_")]

    df["impact_factor_sum"] = (df[impact_cols] == "yes").sum(axis=1).astype("int8")
    return df

# ---------- helper for engineering a multi-location feature ----------

def add_any_firearm(df):
    """
    creating any_firearm = 1 if firearm aimed or fired, else 0
    """
    df = df.copy()
    df["any_firearm"] = (
        (df["firearms_aimed"] == "yes") |
        (df["firearms_fired"] == "yes")
    ).astype(int)
    return df

# ---------- helper for multi-location feature ----------

def add_multi_location(df):
    """
    creating multi_location = 1 if any location columns are flagged, else 0
    """
    df = df.copy()
    loc_cols = [c for c in df.columns if c.startswith("location_")]
    df["multi_location"] = ((df[loc_cols] == "yes").sum(axis=1) > 1).astype("int8")
    return df

# ---------- Helper for engineering Crowd × Weapon interaction ----------

def add_crowd_weapon_interaction(df):
    """
    creating crowd_weapon_interaction = 1 if crowd present and weapon used, else 0
    im feature engineering a feature that captures the interaction between
    crowd presence and weapon possession.
    """
    df = df.copy()
    df["crowd_weapon"] = (
        (df["impact_factor_crowd"] == "yes") &
        (df["impact_factor_possession_weapon"] == "yes")
    ).astype(int)
    return df

# ---------- Helper for engineering Data quality risk count, Missing details can itself signal weird, chaotic, severe incidents ----------

# --------------- row level NA counter ------------------------------
def add_na_count_not_stated(df):
    """
    adding na_count_not_stated:
        number of columns that are NA for this row
        excluding columns that are 100 % NA in that rows year.
    """
    df = df.copy()

    # identifying structural NA columns per year
    structural_na = (
        df.groupby("year").apply(lambda g: g.isna().all())  # bool DF
    )  # index: (year, column)

    # function to count row level NA excluding structural NAs
    def count_row_na(row):
        year = row["year"]
        valid_mask = ~structural_na.loc[year]   
        return row[valid_mask].isna().sum()

    df["na_count_not_stated"] = df.apply(count_row_na, axis=1).astype("int16")
    return df


