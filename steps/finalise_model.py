from zenml.steps import step
from pathlib import Path
import datetime, json, joblib
import pandas as pd
from sklearn.metrics import classification_report, balanced_accuracy_score
from src.modeling import build_model, get_leakage_cols
import numpy as np 
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from src.calibration_plots_fairness_utils import equal_opportunity, equalized_odds


from pycalib.metrics import binary_ECE, binary_MCE
from pycalib.visualisations import plot_reliability_diagram
from src.cf_utils import infer_binary_cols, CAT_BASE, NUM_BASE



@step(enable_cache=False)
def finalise_model(fe_df, version = "v1", tau = 0.044, holdout_year = "2023/24",out_dir = "saved_model"):
    
    df = fe_df.copy()
    # ---- split train/holdout -----------------------------------
    full_train = df[df["year"] != holdout_year].reset_index(drop=True)
    X_full = full_train.drop(columns=get_leakage_cols(full_train))
    y_full = full_train["highrisk"].map({"Minor": 0, "HighRisk": 1}).astype(int)

    test_df = df[df["year"] == holdout_year].reset_index(drop=True)
    X_test  = test_df.drop(columns=get_leakage_cols(test_df))
    y_test  = test_df["highrisk"].map({"Minor": 0, "HighRisk": 1}).astype(int)

    # ---- fit ----------------------------------------------------
    model = build_model(X_full).fit(X_full, y_full)
    
    

    # ---- evaluating at custom threshold ---------------------------
    proba  = model.predict_proba(X_test)[:, 1]  # Get predicted probabilities for the positive class
    
    y_pred = (proba >= tau).astype(int)

    # ---- ensuring that the output directory exists before saving figures ----
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- threshold-free metrics ----------------------------------
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.RocCurveDisplay.html?utm_source
    roc_auc = float(roc_auc_score(y_test, proba))
    pr_auc  = float(average_precision_score(y_test, proba))   # PR-AUC (AP)
    #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html?utm_source
    brier   = float(brier_score_loss(y_test, proba))
    
    #thresholded metric
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    
    

    # https://classifier-calibration.github.io/PyCalib/api/metrics.html#pycalib.metrics.ECE
    
    # ---- calibration (ECE + Brier score + MCE reliability diagram) -----------------
    # --- Calibration with PyCalib (binary) ---

    # 1) flatten and clean
    probs  = np.asarray(proba, dtype=float).ravel()     # P(y=1)
    labels = np.asarray(y_test, dtype=int).ravel()

    good = np.isfinite(probs) & np.isfinite(labels)
    probs, labels = probs[good], labels[good]


    ece_pycalib = float(binary_ECE(labels, probs, bins=10))
    mce_pycalib = float(binary_MCE(labels, probs, bins=10))

    # 3) reliability diagram (needs n x 2 matrix of class probs according to documentation)
    scores_2d = np.column_stack([1.0 - probs, probs])  # columns P(y=0), P(y=1)

    fig = plot_reliability_diagram(
        labels=labels,
        scores=scores_2d,            
        bins=10,
        legend=[f"High-Risk (calibrated)\nECE={ece_pycalib:.3f}\nBrier_Score={brier:.3f}\nMCE={mce_pycalib:.3f}"],
        class_names=["Minor", "High-Risk"], 
        show_histogram=False
    )
    fig.savefig(out / "reliability_pycalib.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


    # ---- plots: ROC + PR -----------------------------------------
    RocCurveDisplay.from_predictions(y_test, proba)
    plt.title(f"ROC (AUC={roc_auc:.3f})")
    plt.savefig(out / "roc_curve.png", bbox_inches="tight")
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_test, proba)
    plt.title(f"PR (AP={pr_auc:.3f})")
    plt.savefig(out / "pr_curve.png", bbox_inches="tight")
    plt.close()

    # ---- thresholded metrics (at tau) ----------------------------
    report  = classification_report(y_test, y_pred, target_names=["Minor", "HighRisk"])
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    print(f"\nClassification report at tau={tau:.3f} (holdout={holdout_year})")
    print(report)
    print("Balanced accuracy:", round(bal_acc, 3))
    
    
    
    
    
# ----------------- Fairness (Fairlearn package/library): Equal Opportunity and Equalised Odds -----------------------
   
    # subgroup FNRs on the holdout fairness slices
    #https://medium.com/data-science/analysing-fairness-in-machine-learning-with-python-96a9ab0d0705

    # ---  Fairness Fairlearn package was used: Equal Opportunity & Equalised Odds ---
    equalised_opportunity_ethnicity = equal_opportunity(
        y_true=y_test,
        y_pred=y_pred,  # thresholded at my  tau
        group_series=test_df["person_perceived_ethnicity"],
    
    )

    equalised_opportunity_gender = equal_opportunity(
        y_true=y_test,
        y_pred=y_pred,
        group_series=test_df["person_perceived_gender"],
    )

    equalised_odds_ethnicity = equalized_odds(
        y_true=y_test,
        y_pred=y_pred,
        group_series=test_df["person_perceived_ethnicity"],

    )

    equalised_odds_gender = equalized_odds(
        y_true=y_test,
        y_pred=y_pred,
        group_series=test_df["person_perceived_gender"],
    )


    # ---- saving fairness results to json ------------------------
    (out / "fairness_fairlearn.json").write_text(
        json.dumps(
            {
                "tau_used": float(tau),
                "equal_opportunity": {
                    "ethnicity": equalised_opportunity_ethnicity,
                    "gender": equalised_opportunity_gender,
                },
                "equalized_odds": {
                    "ethnicity": equalised_odds_ethnicity,
                    "gender": equalised_odds_gender,
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8", # for special characters
    )
 
    
    # ---- saving my report ---------
    #------------------------------
    report_txt = (
        f"tau={tau:.3f}  holdout={holdout_year}\n\n"
        f"{report}\n"
        f"Balanced accuracy: {bal_acc:.3f}\n"
        f"ROC-AUC: {roc_auc:.3f}\n"
        f"PR-AUC: {pr_auc:.3f}\n"
        f"Brier: {brier:.3f}\n"
        f"ECE: {ece_pycalib:.3f}\n"
        f"MCE: {mce_pycalib:.3f}\n"
        f"Macro-F1: {macro_f1:.3f}\n"
        
    )
    (out / f"highrisk_{version}_report.txt").write_text(report_txt, encoding="utf-8")




    # ---- saving my model ---------------------------------------------
    model_path = out / f"highrisk_{version}.pkl"
    joblib.dump(model, model_path)
    
    # ---- selected features (after selector) ----------------------
    feat_all = model.named_steps["prep"].get_feature_names_out()
    support  = model.named_steps["select"].get_support()
    features_selected = np.array(feat_all)[support].tolist()

    # --------- save metadata ------------------------------------------
    meta = {
        "version": version,
        "timestamp": datetime.date.today().isoformat(),
        "tau": float(tau),
        "holdout_year": holdout_year,
        "train_size": int(len(X_full)),
        

        # features before/after selector
        "n_features_after_prep": int(len(feat_all)),
        "n_features_selected": int(len(features_selected)),
        "features_after_prep": feat_all.tolist(),
        "features_selected": features_selected,

        # metrics on holdout
        "balanced_accuracy_holdout": float(bal_acc),
        "roc_auc_holdout": float(roc_auc),
        "pr_auc_holdout": float(pr_auc),
        "brier_holdout": float(brier),
        "ece_holdout": float(ece_pycalib),
        "mce_holdout": float(mce_pycalib),
        "macro_f1_holdout": float(macro_f1),

        # original raw column order expected by the pipeline
        "raw_columns": list(X_full.columns),
    }

    meta_path = out / f"highrisk_{version}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


    print(f"saving model to: {model_path.resolve()}")
    print(f"saved metadata to: {meta_path.resolve()}")
    
    
    
    
    
    
    
    
    #This block packages the exact schema and example rows DiCE needs,
    # so my counterfactuals use the same inputs and the same threshold as the deployed model.
    
    
    # -------------  persist DiCE inputs ------------------------------
    # making sure base frames themselves have unique columns
    
    # this is needed to make the schema stable for counterfactuals
    ## DiCE (and my bridge code) need a fixed, unambiguous set of columns so here im dropping duplicates
    X_full = X_full.loc[:, ~X_full.columns.duplicated()].copy()
    X_test = X_test.loc[:, ~X_test.columns.duplicated()].copy()
    
    

    # building feature groups
    # here i am Defining feature groups that DiCE can understand and edit
    bin_cols_all = infer_binary_cols(X_full) #Calling my helper to auto detec true/false or 0/1 style fields from the training features.
    cat_cols     = [column for column in CAT_BASE if column in X_full.columns] #taking the CAT_BASE list from cf_utils and keeps only those that are in the training features
    # if a column is in the categorical list, im making sure to not r treat it as binary
    bin_cols     = [column for column in bin_cols_all if column not in cat_cols] #removing any overlaps between binary and categorical lists preferring categorical, because it allows missing n other levels in futu
    num_cols     = [column for column in NUM_BASE if column in X_full.columns] #taking the NUM_BASE list from cf_utils and keeps only those that are in the training features

    # keep order but remove duplicates
    #cols_used is the contract the exact raw features ill be giving to DiCE and my bridge    no extras, no duplicates, predictable order.
    cols_used = list(dict.fromkeys([*bin_cols, *cat_cols, *num_cols])) #drops duplicates too while preserving order, as this becomes the schema ill give to dice and ym bridge, its table no duplicates



    # -------- Update meta.json with schema details (so app/steps can align) -------
    # - Stores the final schema (what DiCE and my wrappers should expect) in meta
    # - 'cols_used' is the  source  for column order
    # - here type lists (binary/categorical/numeric) aligned to cols_used
    
    meta.update({
        "cols_used": cols_used,
        "binary_cols": [c for c in bin_cols if c in cols_used],
        "cat_cols":    [c for c in cat_cols if c in cols_used],
        "num_cols":    [c for c in num_cols if c in cols_used],
        #info about temporal coverage in the training slice.
        "years_in_train": sorted(full_train["year"].dropna().unique().tolist()), #list of all years in the training data
    })
    #Streamlit app can load this schema with columns instead of guessing schemas.
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8") #overwriting the old meta with new version including schema info



    # --------  dice_train raw training features + binary target ---------------
    #DiCEs Data interface needs examples from training to infer valid values and distributions.
    dice_train = X_full[cols_used].copy() # so im basically only selecting the agreed schema columns from the full training features
    
    dice_train = dice_train.loc[:, ~dice_train.columns.duplicated()]  # lil safety here: again just in case im removing duplicate columns (shouldn't be any really)
    
    #Storing this snapshot (with the binary target) lets DiCE create believable counterfactuals rooted in data I actually trained on.
    dice_train["highrisk"] = y_full.values.astype(int) #adding binary target needed back by dice for training its internal model
    
    #I used parquet because preserves data types and is fast;
    dice_train.to_parquet(out / "dice_train.parquet", index=False) # Saves a compact data type preserving training snapshot for DiCE.
    #this reason for this is because diCE learns realistic value ranges and categories from this data.



    # --------  holdout_highrisk: ensure at least one query row -----------------
   
    proba_hold = model.predict_proba(X_test)[:, 1] # Prefer true HighRisk rows at tau; if none, take the highest-proba row.
    mask_hr    = (proba_hold >= tau)
    #---------------- Scores the hold-out features and flags rows that are High-Risk at my deployed τ ---------


    # ---------- makes sure theres always a valid, realistic single query row for CF generation -----------
    
    #if I have any High-Risk rows at tau, it take the first one
    #Otherwise it takes the highest probability row as a fallback.
    #always makes sure there’s always a valid, realistic single query row for CF generation
    if mask_hr.any():
        holdout_pool = X_test.loc[mask_hr, cols_used].copy()
        # For determinism, keep the first HighRisk row
        holdout_one  = holdout_pool.iloc[[0]].copy()
    else:
        top_idx = int(np.argmax(proba_hold))
        holdout_one = X_test.loc[[X_test.index[top_idx]], cols_used].copy()
    #------------------------------------------------------------------------------------------------------



    # i made sure to put a final duplicate column guard then i my save my holdout_highrisk as parquet
    holdout_one = holdout_one.loc[:, ~holdout_one.columns.duplicated()]
    holdout_one.to_parquet(out / "holdout_highrisk.parquet", index=False) # Save that one-row DataFrame as a parquet file

    print("saved dice_train.parquet and holdout_highrisk.parquet bra")
    # ============================================================================

    return str(model_path.resolve()) 
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    

