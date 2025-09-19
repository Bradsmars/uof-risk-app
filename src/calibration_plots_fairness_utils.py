# --- Compatibility wrappers to mirror WTTech blog function names ---

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import pandas as pd
from fairlearn.metrics import equal_opportunity_ratio, equal_opportunity_difference, equalized_odds_difference, equalized_odds_ratio, true_positive_rate, false_positive_rate, MetricFrame






# https://fairlearn.org/main/api_reference/generated/fairlearn.metrics.equal_opportunity_ratio.html
def equal_opportunity(y_true, y_pred, group_series, method = "between_groups"):
    """
    Equal Opportunity = TPR parity across groups.
    Uses Fairlearn's scalar metrics and also returns per-group TPR (and FNR=1-TPR).

    using y_pred thresholded 0/1 labels at τ, not probabilities.
    
    aggregation method: worstcase werent chosen becuase were only looking at TPR here.
    i choose "between_groups" becuase it compares groups to each other.
    """
    
    
    groups = pd.Series(group_series, dtype="string").fillna("missing")

    # scalars from Fairlearn (how far groups are from parity)
    eopp_ratio = float(equal_opportunity_ratio(
        y_true=y_true,
        y_pred=y_pred, 
        sensitive_features=groups,
        method=method
    ))
    eopp_diff = float(equal_opportunity_difference(
        y_true=y_true,
        y_pred=y_pred, 
        sensitive_features=groups,
        method=method
    ))

    # per group True postive rates through MetricFrame
    #  encoding fpr was a mistake, but its here now, in methodology ill only report TPR
    metric_frame = MetricFrame(
        metrics={"TPR": true_positive_rate, "FPR": false_positive_rate},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=groups,
    )
   
   # per group TPR/FPR as dictionary
    true_positive_rate_by = metric_frame.by_group["TPR"].astype(float).to_dict()
    false_positive_rate_by = metric_frame.by_group["FPR"].astype(float).to_dict()


    return {
        "equal_opportunity_ratio": eopp_ratio,
        "equal_opportunity_difference": eopp_diff,
        "by_group": {"TPR": true_positive_rate_by, "FPR": false_positive_rate_by},
        "counts": groups.value_counts().astype(int).to_dict(),
    }
    
    
    
#https://fairlearn.org/main/api_reference/generated/fairlearn.metrics.equalized_odds_ratio.html

def equalized_odds(y_true,y_pred,group_series,
    method = "between_groups",   # same default as MetricFrame.ratio()/difference
    agg = "worst_case",          # I choose "worst_case" as 
):
    """
    Equalised Odds summary + breakdown.

    using y_pred thats thresholded 0/1 labels at my τ (not probabilities btw)
    
     returns per-group TPR/FPR for auditability
     
     i choose worst case becuase  - worst_case looks across both TPR and FPR, and across all groups, and then reports the single biggest disparity it finds.
     One bad gap anywhere sets the score.

    Returns
         as a dictionary:
         
          equalized_odds_difference: float
          equalized_odds_ratio: float
          ratios_agg: "worst_case" 
          method: "between_groups"  "to_overall"
          by_group: TPR, FPR
          counts
        
    """
    groups = pd.Series(group_series, dtype="string").fillna("missing")

    # Scalars from Fairlearn
    eo_diff = float(equalized_odds_difference(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=groups,
        method=method,
    ))
    eo_ratio = float(equalized_odds_ratio(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=groups,
        method=method,
        agg=agg,
    ))

    # per group TPR/FPR
    metric_frame = MetricFrame(
        metrics={"TPR": true_positive_rate, "FPR": false_positive_rate},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=groups,
    )
    true_positive_rate_by = metric_frame.by_group["TPR"].astype(float).to_dict()
    false_positive_rate_by = metric_frame.by_group["FPR"].astype(float).to_dict()

    return {
        "equalized_odds_difference": eo_diff,
        "equalized_odds_ratio": eo_ratio,
        "ratios_agg": agg,
        "method": method,
        "by_group": {"TPR": true_positive_rate_by, "FPR": false_positive_rate_by},
        "counts": groups.value_counts().astype(int).to_dict(),
    }

    
    

