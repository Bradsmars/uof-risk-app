"""
Cleaning helpers for the  datasets.

"""



import pandas as pd



class Cleaner_2020_2023:
    """applying all 2020 to 2023 cleaning rules to a raw df."""

    # shared constants
    na_codes = {"not_stated", "", None}
    
    
    bool_mapping = {"yes": True, "no": False, "not_applicable": False}

    # ---------------------- part 2020 to 2023 helpers ------------------------ #
    @staticmethod
    def replace_na(df, columns):
        """replacing Not available tokens with pandas NA in the given columns"""
        df[columns] = df[columns].replace(Cleaner_2020_2023.na_codes, pd.NA)
        return df

    @staticmethod
    def boolean_values(series, mapping):
        """ mapping the strings  bool, keep nulls, return Boolean data type """
        return series.replace(mapping).astype("boolean")
    
    @staticmethod
    def fix_irritant_consistency(df):
        """
        If grouped_irritant_used == 'yes' and grouped_irritant_drawn == 'no',
        im going to change 'drawn' to 'yes' so the pair is logically consistent, t
        this makes sense as you cant use something without drawing it first
        """
        mask = (
            (df["grouped_irritant_used"] == "yes")
            & (df["grouped_irritant_drawn"] == "no")
        )
        df.loc[mask, "grouped_irritant_drawn"] = "yes"
        return df
    
    
    @staticmethod
    def drop_column(df, cols):
        """dropping columns from the dataframe helper"""
        return df.drop(columns=cols, errors="ignore")

    @staticmethod
    def add_officer_injured_no_assault(df):
        """
        creating officer_injured_no_assault feature:
            yes:  officer_injured == "yes"  and  assault flag is False
            no:  otherwise
        """
        df["officer_injured_no_assault"] = (
            (df["officer_injured"] == "yes") & (~df["officer_injured_intentional_assault"])
        ).map({True: "yes", False: "no"})
        return df
    
    

    

    # -------------------------- 2020 to 2023 dataframe cleaners -------------------------- #
    def clean(self, df):
        """returning a new DataFrame with 2020 and 2023 rules applied"""
        df = df.copy()

        # 1) Normaliske NA tokens
        cols = ["officer_injured", "officer_injured_intentional_assault"]
        df = self.replace_na(df, cols)

        # 2) booleanise intentional assault column
        df["officer_injured_intentional_assault"] = self.boolean_values(
            df["officer_injured_intentional_assault"], self.bool_mapping
        )

        # 3) Build the union flag
        df["officer_injured_assault"] = (
            (df["officer_injured"] == "yes")
            | (df["officer_injured_intentional_assault"] == True)
        ).astype("boolean")
        
        df = self.fix_irritant_consistency(df)
        
        #df = self.drop_column(df, ["officer_injured_assault"])
        
        df = self.add_officer_injured_no_assault(df)
        
        df["officer_injured_assault"] = df["officer_injured_assault"].astype("string")

        return df
    
    
    
    
    
class Cleaner2024:
    """ apply all 2024 cleaning rules to a raw df"""

    # ---------- rename map ----------
    RENAME_MAP = {
        "person_age":               "person_perceived_age",
        "person_gender":            "person_perceived_gender",
        "person_ethnicity":         "person_perceived_ethnicity",
        "health_condition":         "person_mental_health_cond",
        "force_name":               "police_force",
        "irritant_spray_drawn":     "grouped_irritant_drawn",
        "irritant_spray_used":      "grouped_irritant_used",
    }

    # ----------- 2024 helpers ------------------------------------------- #
    @staticmethod
    def rename_columns(df, cols):
        """
        i want to rename columns within my 2024 dataframe
        so that theyre consistent with earlier years.
        """
        return df.rename(columns=cols)
    
    @staticmethod
    def fix_health_columns(df):
        """standardising mental and physical health flags"""
        # Making sure the physical column exists
        if "person_physical_health_cond" not in df.columns:
            df["person_physical_health_cond"] = pd.NA

        # Work on a trimmed string view of the mental column
        src = df["person_mental_health_cond"].astype("string").str.strip()

        mental_map = {
            "mental_health_condition":          "yes",
            "physical_mental_health_condition": "yes",
            "physical_health_condition":        "no",
            "no_health_condition":              "no",
        }
        df["person_mental_health_cond"] = (
            src.replace(mental_map).replace({"<NA>": pd.NA})
        ).astype("string")

        physical_map = {
            "physical_health_condition":        "yes",
            "physical_mental_health_condition": "yes",
            "mental_health_condition":          "no",
            "no_health_condition":              "no",
        }
        df["person_physical_health_cond"] = (
            src.replace(physical_map).replace({"<NA>": pd.NA})
        ).astype("string")

        return df
    
    @staticmethod
    def mapping_gender_values(df):
        """ mapping 2024 gender labels to earlier year standards"""
        gender_map = {
            "man":   "male",
            "woman": "female",
            "other": "other",
            "":      pd.NA,
        }
        df["person_perceived_gender"] = (
            df["person_perceived_gender"]
              .astype("string").str.strip().str.lower()
              .replace(gender_map)
        )
        return df
    
    @staticmethod
    def fix_irritant_consistency(df):
        """
        If grouped_irritant_used == 'yes' and grouped_irritant_drawn == 'no',
        then im going toflip 'drawn' to 'yes' so the pair is logically make sense
        """
        mask = (
            (df["grouped_irritant_used"] == "yes")
            & (df["grouped_irritant_drawn"] == "no")
        )
        df.loc[mask, "grouped_irritant_drawn"] = "yes"
        return df
    
    @staticmethod
    def collapse_assault_flag(df):
        """
        replacing the high cardinality officer_injured_assault codes with a clean
        boolean column officer_injured_intentional_assault
        """
        assault_map = {
            "injured_from_assault":     True,
            "assaulted_not_injured":    True,
            "no_assault_not_injured":   False,
            "injured_not_from_assault": False,
            "not_stated":               False,
            "":                         False
        }

        bool_flag = (
            df["officer_injured_assault"]
            .map(assault_map)
            .fillna(False)
            .astype(bool)
        )

        df["officer_injured_intentional_assault"] = bool_flag
        return df
    
    @staticmethod
    def drop_column(df, cols):
        """dropping columns from the dataFrame"""
        return df.drop(columns=cols, errors="ignore")
    
    
    @staticmethod
    def add_officer_injured_no_assault(df):
        """
        creating officer_injured_no_assault:
            yes: officer_injured == "yes"  and  assault flag is False
            no: otherwise
        """
        df["officer_injured_no_assault"] = (
            (df["officer_injured"] == "yes") & (~df["officer_injured_intentional_assault"])
        ).map({True: "yes", False: "no"})
        return df

 # -----------------------  2024 df cleaner -------------------------------------------- #

    def clean(self, df):
        """returning a NEW DF with 2024 rules applied."""
        df = df.copy()

        # 1) Rename columns (pandas skips keys that don’t exist)
        df = self.rename_columns(df, self.RENAME_MAP)
        # 2) Fix health condition columns
        df = self.fix_health_columns(df)

        df = self.mapping_gender_values(df)
        
        df = self.fix_irritant_consistency(df)
        
        df = self.collapse_assault_flag(df)
        
        df = self.add_officer_injured_no_assault(df)
        
        df["officer_injured_assault"] = df["officer_injured_assault"].astype("string")
        
        #df = self.drop_column(df, ["officer_injured_assault"])

        return df

