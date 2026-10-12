# Bankruptcy Prediction Model: Italy & Lithuania Case Study

This repository contains a machine learning project aimed at early detection of corporate financial distress, focusing on a case study of 446 European firms (402 in Italy, 44 in Lithuania) over the fiscal period spanning 2016 to 2018.

## Project Background & Scope
The primary goal is to analyze corporate financial statements to identify patterns leading to bankruptcy. By moving beyond static single-year snapshots, this study focuses on multi-year trends and trajectories to detect deteriorating health early. This allows for a proactive framework for financial distress detection across varied European markets.

## Data Scope and Analytical Framework
The research analyzes the multi-year financial performance based on the Altman model, looking at:
*   **Liquidity:** Working Cap / Total Assets (TA), Total Liab / TA
*   **Profitability:** Retained Earn / TA, EBIT / TA, Net Income / TA
*   **Solvency:** Equity / TA, Debt / Equity
*   **Efficiency:** Asset Turnover, Net Profit Margin, Net Debt / EBIT

### Dynamic Trends Feature Engineering
To capture the momentum of financial distress, we engineered trend-based features:
*   **Recent Trend (Momentum):** Δ₂ = Ratio (2018) - Ratio (2017)
*   **Acceleration (Spiraling):** Acc = Δ₂ - Δ₁
*   **Volatility:** High fluctuations in efficiency and profitability ratios.
*   **Predictivity:** Recent changes are heavily weighted as more predictive than older ones.

## Machine Learning Methodology
The project employs a Voting Ensemble approach combining:
*   **Logistic Regression:** Captures linear coefficients of features to determine the baseline probability of default.
*   **Random Forest:** Uses an ensemble of decision trees to identify complex non-linear relationships in the data.
*   **Ensemble Learning:** Optimized weights (Soft Voting) for each algorithm reduce variance and improve stability.

## Key Risk Drivers
Based on feature importance analysis, the following drivers were identified:

**Drivers that increase risk:**
*   Total Liabilities / Total Assets [Level 2018]
*   Total Liabilities / Total Assets [Δ Trend 2017-2018]
*   Net Debt / EBIT [Δ Trend 2017-2018]

**Drivers that reduce risk:**
*   Shareholders Funds / Total Assets [Level 2018]
*   Retained Earnings / Total Assets [Level 2018]
*   Working Capital / Total Assets [Level 2018]

**Key Vulnerabilities / Red Flags:**
*   Net Income / Assets < 0
*   Sudden drop in Current Ratio
*   (EBIT) trend acceleration downwards

**Mitigation Factors:**
*   High retained earnings
*   Stable efficiency ratios
*   Strong interest coverage (> 3.0)

## Repository Contents
*   `financial.py`: Main Python script containing the feature engineering (building ratios, deltas, and accelerations), model training (Logistic Regression, Random Forest, and Voting Ensemble with GridSearchCV), and evaluation metrics.
*   `challenge_train.xlsx`: Training dataset containing the financial data.
*   `challenge_test_without_target_.xlsx`: Test dataset for predictions.


## Final Conclusions
*   **Ensemble Success:** The combination of Logistic Regression and Random Forest provided stable results across regional datasets.
*   **Dynamic Superiority:** Trend-based features (Δ) often outperformed static single-year snapshots in predicting eventual bankruptcy.
*   **Operational Impact:** The model allows financial institutions to choose a customized decision boundary threshold to balance false positives and false negatives according to their risk tolerance.
