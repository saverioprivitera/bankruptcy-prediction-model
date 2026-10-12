import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

import random
import numpy as np
np.random.seed(42)
random.seed(42)


def safe_div(numerator, denominator):
    """
    Safely divides two arrays, returning 0 where the denominator is 0.
    """
    return np.where(denominator != 0, numerator / denominator, 0)

def build_ratios(df):

    """

    Builds the full feature matrix:

      Level ratios for 2016, 2017, 2018  (each economic dimension)

       Δ (first difference) 2016→2017 and 2017→2018

       2-year change 2016→2018

       Acceleration: does the trend itself accelerate? (Δ of Δ)

       Volatility: std of the ratio across the 3 years

    """

    d = df.copy()

    r = pd.DataFrame()

    r["firm_id"] = d["firm_id"]



    # Helper: pull a named item for a given year

    def col(name, yr):

        return d[f"{name} th EUR {yr}"]



    YEARS = ["2016", "2017", "2018"]



    #  collecting raw series for every year 

    ta   = {y: col("Total assets", y)                             for y in YEARS}

    curr = {y: col("Current assets", y)                           for y in YEARS}

    cl   = {y: col("Current liabilities", y)                      for y in YEARS}

    sf   = {y: col("Shareholders funds", y)                       for y in YEARS}

    tl   = {y: d[f"Total shareholders' funds and liabilities th EUR {y}"] - sf[y] for y in YEARS}

    ebit = {y: col("Operating profit (loss) [EBIT]", y)           for y in YEARS}

    ni   = {y: col("Profit (loss) for the period [Net income]", y) for y in YEARS}

    sales= {y: col("Sales", y)                                    for y in YEARS}

    cash = {y: col("Of which cash and cash equivalent", y)        for y in YEARS}

    dlt  = {y: col("Long term debt", y)                           for y in YEARS}

    dst  = {y: col("Loans & short-term debt", y)                  for y in YEARS}

    fexp = {y: col("Financial expenses", y)                       for y in YEARS}

    oth_sf={y: col("Other shareholders funds", y)                 for y in YEARS}

    gp   = {y: col("Gross profit", y)                             for y in YEARS}

    cred = {y: col("Creditors", y)                                for y in YEARS}

    debt = {y: dlt[y] + dst[y]                                    for y in YEARS}

    wc   = {y: curr[y] - cl[y]                                    for y in YEARS}



   
    # LIQUIDITY

    wc_ta      = {y: safe_div(wc[y],   ta[y])   for y in YEARS}

    cash_ta    = {y: safe_div(cash[y], ta[y])   for y in YEARS}

    curr_ratio = {y: safe_div(curr[y], cl[y])   for y in YEARS}



    # PROFITABILITY

    ebit_ta    = {y: safe_div(ebit[y], ta[y])   for y in YEARS}

    ni_ta      = {y: safe_div(ni[y],   ta[y])   for y in YEARS}

    retained_ta= {y: safe_div(oth_sf[y], ta[y]) for y in YEARS}

    gp_sales   = {y: safe_div(gp[y],  sales[y]) for y in YEARS}   # Gross margin

    net_profit_margin = {y: safe_div(ni[y], sales[y]) for y in YEARS}



    # LEVERAGE

    tl_ta      = {y: safe_div(tl[y],  ta[y])   for y in YEARS}

    sf_ta      = {y: safe_div(sf[y],  ta[y])   for y in YEARS}

    debt_ta    = {y: safe_div(debt[y],ta[y])   for y in YEARS}

    debt_equity = {y: safe_div(debt[y], sf[y]) for y in YEARS}



    # FINANCIAL BURDEN

    debt_nz    = {y: debt[y].replace(0, np.nan) for y in YEARS}

    int_cov    = {y: safe_div(ebit[y], fexp[y]) for y in YEARS}

    fin_exp_dbt= {y: safe_div(fexp[y], debt_nz[y]) for y in YEARS}

    net_debt_ebit = {y: safe_div(debt[y] - cash[y], ebit[y]) for y in YEARS}




    # EFFICIENCY

    sales_ta   = {y: safe_div(sales[y], ta[y])  for y in YEARS}

    cred_sales = {y: safe_div(cred[y],  sales[y]) for y in YEARS}  # Creditor days proxy



   

    ratio_series = {

        "X1": wc_ta,

        "X2": retained_ta,

        "X3": ebit_ta,

        "X4": sf_ta,

        "X5": sales_ta,

        "X6": tl_ta,

        "X7": debt_equity,

        "X8": net_profit_margin,

        "X9": net_debt_ebit,

        "X10": ni_ta,

    }



    for name, series in ratio_series.items():

        for y in YEARS:

            r[f"{name}_{y}"] = series[y]




    # For each ratio we compute:
    #   Δ_a  = value_17 - value_16 (early change)

    #   Δ_b  = value_18 - value_17  (recent change)  

    #   Δ_2y = value_18 - value_16  (full 2-year change)

    #   ACC  = Δ_b - Δ_a            (acceleration)

    #   VOL  = std across 3 years  (instability signal)



    for name, series in ratio_series.items():

        v16 = np.array(series["2016"], dtype=float)

        v17 = np.array(series["2017"], dtype=float)

        v18 = np.array(series["2018"], dtype=float)



        delta_a  = v17 - v16

        delta_b  = v18 - v17

        delta_2y = v18 - v16

        acc      = delta_b - delta_a

        vol      = np.nanstd(np.stack([v16, v17, v18], axis=1), axis=1)



        r[f"{name}_d1"]  = delta_a     # 2016→2017 change

        r[f"{name}_d2"]  = delta_b     # 2017→2018 change

        r[f"{name}_d2y"] = delta_2y    # full 2 year change

        r[f"{name}_acc"] = acc         # acceleration (Δ of Δ)

        r[f"{name}_vol"] = vol         # volatility across years



    

    # Sales growth rate 

    r["sales_growth_1y"] = safe_div(sales["2018"] - sales["2017"], sales["2017"].abs())

    r["sales_growth_2y"] = safe_div(sales["2018"] - sales["2016"], sales["2016"].abs())



    # Asset growth 

    r["ta_growth_1y"]    = safe_div(ta["2018"] - ta["2017"], ta["2017"].abs())



    # Equity erosion
    
    r["sf_negative_18"]  = (sf["2018"] < 0).astype(float)

    r["sf_negative_17"]  = (sf["2017"] < 0).astype(float)

    r["sf_negative_any"] = (
        (sf["2016"] < 0) | (sf["2017"] < 0) | (sf["2018"] < 0)
    ).astype(float)



    # Consecutive loss flag

    loss16 = (ni["2016"] < 0).astype(int)

    loss17 = (ni["2017"] < 0).astype(int)

    loss18 = (ni["2018"] < 0).astype(int)

    r["consec_loss_2y"]  = ((loss17 + loss18) == 2).astype(float)

    r["consec_loss_3y"]  = ((loss16 + loss17 + loss18) == 3).astype(float)

    r["loss_count"]      = (loss16 + loss17 + loss18).astype(float)



    # Interest coverage 

    r["intcov_lt1_18"]   = (int_cov["2018"] < 1).astype(float)

    r["intcov_lt1_any"]  = (
        (int_cov["2016"] < 1) | (int_cov["2017"] < 1) | (int_cov["2018"] < 1)
    ).astype(float)



    return r

train = pd.read_excel('challenge_train.xlsx')

#  defining the feature list
train_r = build_ratios(train)

features_levels = [f'X{i}_2018' for i in range(1, 11)]
features_deltas = [f'X{i}_d2' for i in range(1, 11)]
final_features_list = features_levels + features_deltas

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer

# Data Preparation
X = train_r[final_features_list]
y = train['target']

#splitting 80% for training-CV and 20% for final testing
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# Impute and scale
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(imputer.fit_transform(X_train_p))
X_test_scaled = scaler.transform(imputer.transform(X_test_p))

print(f"Training size: {X_train_scaled.shape[0]}, Test size: {X_test_scaled.shape[0]}")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#Hyperparameter tuning using Cross-validation

#Logistic regression tuning
lr_param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs']
}
lr_grid = GridSearchCV(LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
                       lr_param_grid, cv=5, scoring='f1')
lr_grid.fit(X_train_scaled, y_train_p)
best_lr = lr_grid.best_estimator_

#Random forest tuning
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8],
    'min_samples_leaf': [5, 10]
}
rf_grid = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42),
                       rf_param_grid, cv=5, scoring='f1')
rf_grid.fit(X_train_scaled, y_train_p)
best_rf = rf_grid.best_estimator_

print(f"Best LR params: {lr_grid.best_params_}")
print(f"Best RF params: {rf_grid.best_params_}")

 
#Feature importance visualization 

import re
import matplotlib.pyplot as plt
import pandas as pd

financial_names = {
    "X1": "Working Capital / Total Assets",
    "X2": "Retained Earnings / Total Assets",
    "X3": "EBIT / Total Assets",
    "X4": "Shareholders Funds / Total Assets",
    "X5": "Sales / Total Assets",
    "X6": "Total Liabilities / Total Assets",
    "X7": "Total Debt / Shareholders Funds",
    "X8": "Net Profit Margin (Net Income / Sales)",
    "X9": "Net Debt / EBIT",
    "X10": "Net Income / Total Assets"
}

suffix_names = {
    "2016": "Level 2016",
    "2017": "Level 2017",
    "2018": "Level 2018",
    "d1": "Δ Trend 2016->2017",
    "d2": "Δ Trend 2017->2018",
    "d2y": "Δ 2-Year Trend (2016->2018)",
    "acc": "Acceleration (Δ of Trend)",
    "vol": "Volatility (3-Year Std Dev)"
}

extra_names = {
    "sales_growth_1y": "1-Year Sales Growth (2017->2018)",
    "sales_growth_2y": "2-Year Sales Growth (2016->2018)",
    "ta_growth_1y": "Total Assets Growth (2017->2018)",
    "sf_negative_18": "Negative Shareholders Funds 2018",
    "sf_negative_17": "Negative Shareholders Funds 2017",
    "sf_negative_any": "Negative Shareholders Funds (Any Year)",
    "consec_loss_2y": "2 Consecutive Years of Losses",
    "consec_loss_3y": "3 Consecutive Years of Losses",
    "loss_count": "Number of Years with Losses",
    "intcov_lt1_18": "Interest Coverage < 1 (2018)"
}

readable_names_list = []
for feat in X_train_p.columns:
    if feat in extra_names:
        readable_names_list.append(extra_names[feat])
    else:
        match = re.match(r'^(X\d+)_(.+)$', feat)
        if match:
            code, suff = match.groups()
            base = financial_names.get(code, code)
            suffix_eng = suffix_names.get(suff, suff)
            readable_names_list.append(f"{base} [{suffix_eng}]")
        else:
            readable_names_list.append(feat)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

# Logistic Regression Chart (Coefficients)
lr_coefs = pd.Series(best_lr.coef_[0], index=readable_names_list).sort_values()
lr_coefs.plot(kind='barh', ax=ax1, color='skyblue')
ax1.set_title('Logistic Regression Coefficients', fontsize=14, fontweight='bold', pad=15)
ax1.tick_params(axis='y', labelsize=10)
ax1.axvline(0, color='black', linestyle='--', linewidth=0.8)

# Random Forest Chart (Feature Importance)
rf_importances = pd.Series(best_rf.feature_importances_, index=readable_names_list).sort_values()
rf_importances.plot(kind='barh', ax=ax2, color='salmon')
ax2.set_title('Random Forest Feature "Importance"', fontsize=14, fontweight='bold', pad=15)
ax2.tick_params(axis='y', labelsize=10)

plt.tight_layout()
plt.show()

from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV

# 6. Optimize Ensemble Weights

# Defining the Votingclassifier 
ensemble_model = VotingClassifier(
    estimators=[('lr', best_lr), ('rf', best_rf)],
    voting='soft', # Use soft voting for probability averaging
    weights=[0.5, 0.5] # Initial weights, will be optimized
)

# Define a grid of weights to search. Weights must sum to 1.
# We'll vary the weight for LR and RF will be 1 - weight_lr
param_grid = {
    'weights': [
        [w, 1-w] for w in np.arange(0.1, 1.0, 0.1)
    ]
}

#  find the best weights for the ensemble
grid_search_ensemble = GridSearchCV(
    estimator=ensemble_model,
    param_grid=param_grid,
    cv=5, # Use 5-fold cross-validation
    scoring='f1', # Optimize for F1 score
    verbose=1,
    n_jobs=-1 # Use all available CPU cores
)


grid_search_ensemble.fit(X_train_scaled, y_train_p)

# Getting the best ensemble model and its weights
best_ensemble_model = grid_search_ensemble.best_estimator_
best_weights = grid_search_ensemble.best_params_['weights']

print(f"Best ensemble weights: {best_weights}")
print(f"Best F1 score from GridSearchCV: {grid_search_ensemble.best_score_:.4f}")


# getting probabilities from the best ensemble model
optimized_ensemble_probs = best_ensemble_model.predict_proba(X_test_scaled)[:, 1]

# Using the previously found best threshold 
THRESHOLD = 0.575
optimized_ensemble_preds = (optimized_ensemble_probs >= THRESHOLD).astype(int)

print(f"\n--- Ensemble Performance (Optimized Weights, Threshold {THRESHOLD}) ---")
print(confusion_matrix(y_test_p, optimized_ensemble_preds))
print(classification_report(y_test_p, optimized_ensemble_preds))
print(f"ROC AUC Score: {roc_auc_score(y_test_p, optimized_ensemble_probs):.4f}")

#  confusion matrix 
cm_optimized = confusion_matrix(y_test_p, optimized_ensemble_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_optimized, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Non-Bankrupt', 'Bankrupt'],
            yticklabels=['Non-Bankrupt', 'Bankrupt'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Optimized Ensemble Confusion Matrix (Threshold {THRESHOLD})')
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

thresholds_grid = np.linspace(0, 1, 200)
total_misclassified = []
false_positives = []
false_negatives = []

# Computing the total errors for each THRESHOLD
for t in thresholds_grid:
    preds = (optimized_ensemble_probs >= t).astype(int)

    # False positives
    fp = np.sum((preds == 1) & (y_test_p == 0))
    # False negatives
    fn = np.sum((preds == 0) & (y_test_p == 1))

    false_positives.append(fp)
    false_negatives.append(fn)
    total_misclassified.append(fp + fn)

# we calculate precision and recall
precision, recall, pr_thresholds = precision_recall_curve(
    y_test_p, optimized_ensemble_probs
)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Precision-Recall curve
ax1.plot(pr_thresholds, precision[:-1], label="Precision", color="tab:red", lw=2.5)
ax1.plot(pr_thresholds, recall[:-1], label="Recall", color="tab:grey", lw=2.5)
ax1.set_title("Precision and Recall vs. Threshold", fontsize=13, pad=10)
ax1.set_xlabel("Threshold", fontsize=11)
ax1.set_ylabel("Score", fontsize=11)
ax1.set_xlim([-0.05, 1.05])
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="center right", fontsize=10)

# False positives-False negative graph
ax2.plot(thresholds_grid, false_positives, label="False positives", color="tab:red", linestyle="-", lw= 2.5)
ax2.plot(thresholds_grid, false_negatives, label="False negatives ", color="tab:grey", linestyle="-", lw= 2.5)
ax2.set_title("Misclassified examples vs. Threshold", fontsize=13, pad=10)
ax2.set_xlabel("Threshold", fontsize=11)
ax2.set_ylabel("Number of companies", fontsize=11)
ax2.set_xlim([-0.05, 1.05])
ax2.legend(loc="center right", fontsize=10)

plt.tight_layout()

plt.savefig("analisi_soglie_ensemble.png", dpi=300)

TEST_PATH = 'challenge_test_without_target_.xlsx'
test = pd.read_excel(TEST_PATH)

test_r = build_ratios(test)

# Selecting the same features used for training
X_test_final = test_r[final_features_list]

X_test_final_scaled = scaler.transform(imputer.transform(X_test_final))

# Get probabilities from the best ensemble model on the final test set
final_ensemble_probs = best_ensemble_model.predict_proba(X_test_final_scaled)[:, 1]

# Applying the optimized threshold to get final predictions
final_predictions = (final_ensemble_probs >= THRESHOLD).astype(int)

submission_df = pd.DataFrame({'firm_id': test['firm_id'], 'predicted_label': final_predictions})

submission_output_path = 'submission_optimized_ensemble.xlsx'
submission_df.to_excel(submission_output_path, index=False)

print(f"File di sottomissione generato: {submission_output_path}")
print(submission_df.head())