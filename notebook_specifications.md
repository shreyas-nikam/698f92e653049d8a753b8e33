
# Model Validation and Performance Degradation Analysis for Financial Models

## Introduction: The Portfolio Manager's Dilemma

**Persona:** Sarah Chen, a seasoned CFA Charterholder and Portfolio Manager at "Alpha Capital Investments."

**Organization:** Alpha Capital Investments, an institutional asset manager with a strong focus on quantitative strategies and rigorous risk management.

Sarah is responsible for a multi-billion dollar equity portfolio. Her team frequently develops and deploys quantitative models for investment decisions, such as equity selection and portfolio optimization. Recently, a new equity factor model, developed by a junior quant, showed impressive $R^2$ values during in-sample testing. However, Sarah, with her years of experience and deep understanding of financial markets, knows that in-sample performance can be a deceptive illusion. She's concerned about the "overfitting gap"—the difference between apparent and true model performance—which can lead to models that look great on paper but fail catastrophically during real-world market stress or regime changes.

Her primary objective is to thoroughly validate this new model to ensure it will perform reliably in forward-looking scenarios and to identify potential "ticking time bombs" that could erode capital. This rigorous validation is not just good practice; it's a professional and ethical obligation under CFA Standard V(A) (Diligence and Reasonable Basis) and aligns with the firm's model risk management (SR 11-7) protocols. This notebook walks through the step-by-step validation process Sarah employs.

---

### Setup: Installing and Importing Libraries

Before we begin our validation journey, we need to install the necessary Python libraries.

```python
!pip install pandas numpy scikit-learn matplotlib seaborn yfinance
```

### Importing Required Dependencies

Next, we import all the libraries crucial for data handling, model building, cross-validation, and visualization.

```python
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit, learning_curve, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import datetime as dt
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100
```

---

## 1. The Peril of Random Splits: Unveiling the Overfitting Gap

### Story + Context + Real-World Relevance

Sarah has received the preliminary report for the new equity factor model. The junior quant proudly highlighted an $R^2$ of $0.12$ on a randomly split test set. However, Sarah immediately raises an eyebrow. Financial time-series data is fundamentally different from typical i.i.d. (independent and identically distributed) data, a core assumption often violated in standard machine learning. Stock returns, for instance, exhibit temporal dependence; today's market conditions often influence tomorrow's. A random train/test split can inadvertently place future data points into the training set and past data points into the test set (or vice-versa, creating overlapping information windows), severely inflating performance metrics due to **information leakage** and **look-ahead bias**. This leads to an **overfitting gap** where apparent performance far exceeds true out-of-sample capability.

Sarah knows that for financial data, preserving the temporal order is paramount. She decides to demonstrate this critical pitfall by comparing the model's performance using a random split versus a strict temporal split.

The standard $R^2$ (coefficient of determination) is calculated as:
$$ R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2} $$
where $y_i$ is the actual value, $\hat{y}_i$ is the predicted value, and $\bar{y}$ is the mean of the actual values.

For financial time series, using random splits often leads to an artificially high $R^2$ value because:
1.  **Temporal Dependence:** Future information "leaks" into the training set, especially if the target variable involves future returns (e.g., 1-month forward returns).
2.  **Non-stationarity:** Market regimes change, and models trained on specific periods might not generalize to others. Random splits mix these regimes.

By comparing $R^2$ from random vs. temporal splits, Sarah quantifies the "overfitting gap," a critical first step to understanding the model's true predictive power. A significant gap indicates severe overfitting.

```python
# --- Data Generation for Demonstration ---
# We'll simulate a simple financial time-series dataset.
# Features (X) will be slightly lagged and correlated with the target (y).
np.random.seed(42)
n_samples = 252 * 10 # 10 years of daily data
dates = pd.date_range(start='2010-01-01', periods=n_samples, freq='B')

# Simulate some financial factors (e.g., momentum, value)
factors = pd.DataFrame(np.random.randn(n_samples, 5), index=dates, columns=[f'Factor_{i}' for i in range(1, 6)])
factors = factors.cumsum() # Make them more "time-series like"

# Simulate an underlying "true" process for target variable
true_signal = 0.5 * factors['Factor_1'].shift(1) + 0.3 * factors['Factor_2'].shift(2) - 0.2 * factors['Factor_3'].shift(1)
# Add some noise and make it a return series
y = (true_signal + np.random.randn(n_samples) * 0.5).diff().fillna(0)
y.name = 'Target_Return'

# Combine features and target, ensuring alignment by date
data = pd.concat([factors, y], axis=1).dropna()
X = data.drop('Target_Return', axis=1)
y = data['Target_Return']
dates_aligned = data.index

print(f"Dataset shape: X={X.shape}, y={y.shape}")
print(f"Data period: {dates_aligned.min().strftime('%Y-%m-%d')} to {dates_aligned.max().strftime('%Y-%m-%d')}")

# Define a simple RandomForestRegressor model
model_config = {'n_estimators': 100, 'max_depth': 5, 'random_state': 42}
model = RandomForestRegressor(**model_config)

def evaluate_split_strategy(X, y, dates, model_config):
    """
    Compares random vs. temporal splitting for a given model configuration.
    """
    model_random = RandomForestRegressor(**model_config)
    model_temporal = RandomForestRegressor(**model_config)

    # 1. Random Split
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y, test_size=0.25, random_state=42)
    model_random.fit(X_train_r, y_train_r)
    r2_random = model_random.score(X_test_r, y_test_r)

    # 2. Temporal Split
    split_idx = int(len(X) * 0.75)
    X_train_t, X_test_t = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_t, y_test_t = y.iloc[:split_idx], y.iloc[split_idx:]
    model_temporal.fit(X_train_t, y_train_t)
    r2_temporal = model_temporal.score(X_test_t, y_test_t)

    print("\n--- Model Performance Comparison (Random vs. Temporal Split) ---")
    print(f"R2 with RANDOM split: {r2_random:.4f}")
    print(f"R2 with TEMPORAL split: {r2_temporal:.4f}")

    overfitting_gap = r2_random - r2_temporal
    print(f"Overfitting Gap (Random R2 - Temporal R2): {overfitting_gap:.4f}")
    if overfitting_gap > 0.05: # A heuristic threshold
        print("\nWarning: Significant overfitting gap detected! Random splitting artificially inflated performance.")

    return r2_random, r2_temporal, overfitting_gap

r2_random, r2_temporal, overfitting_gap = evaluate_split_strategy(X, y, dates_aligned, model_config)
```

### Explanation of Execution

The output clearly shows a discrepancy between the $R^2$ achieved with a random split and that with a temporal split. The `Overfitting Gap` quantifies this difference. For Sarah, this numerical gap is a visceral demonstration of the danger of relying on standard, non-time-series aware validation. A model might appear to explain $12\%$ of the variance using a random split, but perhaps only $2\%$ when validated correctly by respecting temporal order. This insight confirms her initial suspicions and emphasizes the need for more sophisticated time-series validation techniques. It’s the first step in identifying if a model is a "ticking time bomb."

---

## 2. Simulating Real-World Deployment with Expanding Window Cross-Validation

### Story + Context + Real-World Relevance

Having established the limitations of simple train/test splits, Sarah now wants to simulate how the model would perform in a real-time deployment scenario. In practice, models are often retrained periodically (e.g., monthly or quarterly) on all available historical data up to that point. This approach, known as **Expanding Window Walk-Forward Cross-Validation**, mirrors this process. For each fold, the model trains on an ever-growing dataset and then evaluates its performance on the subsequent, unseen period. This provides a more realistic assessment of out-of-sample performance over a sequence of future periods.

The **Expanding Window** formulation for fold $k$ is given by:
$$ D^{(k)}_{\text{train}} = \{(x_t, y_t) : t = 1, \dots, T_k\} $$
$$ D^{(k)}_{\text{test}} = \{(x_t, y_t) : t = T_k + 1, \dots, T_k + h\} $$
where $T_k = T_{\min} + (k-1)h$, $T_{\min}$ is the minimum initial training size, and $h$ is the test window length.
The aggregate performance estimate over $K$ folds is:
$$ P = \frac{1}{K} \sum_{k=1}^{K} L(f^{(k)}, D^{(k)}_{\text{test}}) $$
where $f^{(k)}$ is the model trained on $D^{(k)}_{\text{train}}$ and $L$ is the performance metric (e.g., $R^2$, AUC).

```python
def expanding_window_cv(X, y, dates_index, model_config, n_splits=5):
    """
    Performs expanding window walk-forward cross-validation.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []
    
    print(f"\n--- Expanding Window Walk-Forward Cross-Validation (n_splits={n_splits}) ---")
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Ensure time-series order
        train_start, train_end = dates_index[train_idx.min()], dates_index[train_idx.max()]
        test_start, test_end = dates_index[test_idx.min()], dates_index[test_idx.max()]
        
        model_fold = RandomForestRegressor(**model_config)
        model_fold.fit(X_train, y_train)
        r2_fold = model_fold.score(X_test, y_test)
        
        fold_results.append({
            'Fold': fold + 1,
            'Train_Start': train_start.strftime('%Y-%m-%d'),
            'Train_End': train_end.strftime('%Y-%m-%d'),
            'Test_Start': test_start.strftime('%Y-%m-%d'),
            'Test_End': test_end.strftime('%Y-%m-%d'),
            'R2': r2_fold,
            'N_Train': len(train_idx),
            'N_Test': len(test_idx)
        })
        print(f"Fold {fold+1}: Train from {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}, Test from {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')} -> R2: {r2_fold:.4f}")

    results_df = pd.DataFrame(fold_results)
    mean_r2 = results_df['R2'].mean()
    std_r2 = results_df['R2'].std()
    print(f"\nMean CV R2 (Expanding Window): {mean_r2:.4f} +/- {std_r2:.4f}")
    return results_df, mean_r2, std_r2

expanding_results, expanding_mean_r2, expanding_std_r2 = expanding_window_cv(X, y, dates_aligned, model_config, n_splits=5)
```

### Explanation of Execution

The table above details the performance of the model across different expanding window folds. Each fold's $R^2$ gives Sarah a snapshot of the model's predictive power in a specific future period, trained on all data available up to that point. The `Mean CV R2` and its standard deviation (`Std CV R2`) provide an overall picture of the model's average out-of-sample performance and its variability. A low, stable $R^2$ across folds is generally preferred over a high, volatile one. This gives Sarah a much more realistic baseline performance compared to the single, potentially inflated random split $R^2$. She notes down the `Mean CV R2` as a crucial metric for the model's true effectiveness.

---

## 3. Adapting to Market Dynamics with Sliding Window Cross-Validation

### Story + Context + Real-World Relevance

While expanding windows are good for processes assumed to be relatively stable, Sarah knows that financial markets are often non-stationary. Older data might become irrelevant or even detrimental if the underlying data-generating process has changed (e.g., after a major market crisis or technological shift). In such cases, a **Sliding Window Walk-Forward Cross-Validation** approach is more appropriate. Here, the training set size remains fixed, effectively "sliding" along the time series, only using the most recent `W` periods for training. This allows the model to adapt more quickly to recent market dynamics but comes at the cost of discarding potentially useful older data and reducing training sample size.

The **Sliding Window** formulation for fold $k$ replaces the growing training set with a fixed-size window:
$$ D^{(k)}_{\text{train}} = \{(x_t, y_t) : t = T_k - W + 1, \dots, T_k\} $$
where $W$ is the fixed window width. This discards old data, adapting to non-stationarity but losing sample size. Sarah needs to visualize the out-of-sample $R^2$ over time to understand model stability.

```python
def sliding_window_cv(X, y, dates_index, model_config, train_window_size_months=60, test_window_size_months=12):
    """
    Performs sliding window walk-forward cross-validation.
    train_window_size_months: fixed number of months for training
    test_window_size_months: fixed number of months for testing (step size)
    """
    # Convert months to approximate business days for simplicity (21 business days per month)
    train_window_size = train_window_size_months * 21
    test_window_size = test_window_size_months * 21

    sliding_results = []
    
    print(f"\n--- Sliding Window Walk-Forward Cross-Validation (Train Window: {train_window_size_months} months, Test Window: {test_window_size_months} months) ---")

    # Iterate through the data to create sliding windows
    # Start loop such that the first training window is at least train_window_size
    # and there's enough data for at least one test window
    for i in range(train_window_size, len(X) - test_window_size + 1, test_window_size):
        train_start_idx = i - train_window_size
        train_end_idx = i
        test_start_idx = i
        test_end_idx = i + test_window_size

        if test_end_idx > len(X):
            test_end_idx = len(X)

        X_train, X_test = X.iloc[train_start_idx:train_end_idx], X.iloc[test_start_idx:test_end_idx]
        y_train, y_test = y.iloc[train_start_idx:train_end_idx], y.iloc[test_start_idx:test_end_idx]
        
        if len(y_test) == 0: # Handle cases where the last test window is too small
            continue

        model_fold = RandomForestRegressor(**model_config)
        model_fold.fit(X_train, y_train)
        r2_fold = model_fold.score(X_test, y_test)

        sliding_results.append({
            'Test_Period_Start': dates_index[test_start_idx],
            'Test_Period_End': dates_index[test_end_idx-1] if test_end_idx > 0 else dates_index[test_start_idx],
            'R2': r2_fold,
            'N_Train': len(X_train),
            'N_Test': len(X_test)
        })
        print(f"Test Period: {dates_index[test_start_idx].strftime('%Y-%m-%d')} to {dates_index[test_end_idx-1].strftime('%Y-%m-%d')} -> R2: {r2_fold:.4f}")

    sliding_df = pd.DataFrame(sliding_results)
    
    # Plotting sliding R2 over time
    plt.figure(figsize=(14, 7))
    plt.plot(sliding_df['Test_Period_Start'], sliding_df['R2'], 'b-o', markersize=4, label='Out-of-Sample R2')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Zero R2 Baseline')
    plt.xlabel('Test Period Start')
    plt.ylabel('Out-of-Sample R2')
    plt.title('Sliding-Window Performance Over Time')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    mean_r2_sliding = sliding_df['R2'].mean()
    std_r2_sliding = sliding_df['R2'].std()
    print(f"\nMean CV R2 (Sliding Window): {mean_r2_sliding:.4f} +/- {std_r2_sliding:.4f}")
    return sliding_df, mean_r2_sliding, std_r2_sliding

sliding_results_df, sliding_mean_r2, sliding_std_r2 = sliding_window_cv(X, y, dates_aligned, model_config, train_window_size_months=60, test_window_size_months=12)
```

### Explanation of Execution

The plot illustrates how the model's $R^2$ performance varies over time when using a fixed-size training window. Sarah can visually inspect periods where the model performs better or worse. This helps her assess the model's stability and adaptiveness to changing market conditions. If the $R^2$ drops significantly during known periods of market stress, it's a red flag. Compared to the expanding window, the sliding window provides insights into the model's ability to remain relevant in a non-stationary environment. The choice between expanding and sliding windows is a financial judgment: expanding windows assume more stationarity and leverage more data, while sliding windows prioritize recency and adapt to non-stationarity.

---

## 4. Mitigating Information Leakage with Purged and Embargo Cross-Validation

### Story + Context + Real-World Relevance

Sarah, as a diligent CFA, understands that financial targets often involve overlapping forward returns. For example, if the target $y_t$ is the return from time $t$ to $t+12$ months, then $y_t$ and $y_{t+1}$ share $11$ months of return data. Standard `TimeSeriesSplit` does not account for this, leading to significant **information leakage** when samples in the training set overlap with samples in the test set. This type of leakage can dramatically inflate performance metrics, even in walk-forward validation.

To combat this, Sarah employs **Purged and Embargo Cross-Validation** (as popularized by Lopez de Prado).
*   **Purge:** Remove training samples that overlap with the test sample's target observation period. If $y_t$ is a $T_p$-period forward return, then any training observation $(x_s, y_s)$ where $s$ is within $T_p$ periods of the test set's start should be removed from the training set.
    $$ D^{\text{purged}}_{\text{train}} = \{(x_t, y_t) \in D_{\text{train}} : t \leq T_k - T_p\} $$
*   **Embargo:** Introduce a "buffer" period immediately following the training set and before the test set. This ensures that information from the test set does not inadvertently "leak back" into future training sets through serial correlation or other indirect means.
    $$ D^{\text{embargo}}_{\text{test}} = \{(x_t, y_t) \in D_{\text{test}} : t > T_k + T_e + 1\} $$
Typical choices are $T_p = \text{label horizon}$ (e.g., 12 months for 12-month forward returns) and $T_e = 1-3 \text{ months}$ as an additional buffer. Together, these ensure zero information leakage. Sarah wants to quantify the `Leakage Inflation` by comparing standard walk-forward CV $R^2$ to purged and embargoed CV $R^2$.

```python
def purged_embargo_walk_forward_cv(X, y, dates_index, model_config, n_splits=5, purge_window_months=12, embargo_window_months=1):
    """
    Performs walk-forward cross-validation with purging and embargoing to prevent data leakage.
    purge_window_months: Duration (in months) of the forward label (e.g., 12 for 12-month forward returns).
                         Training samples within this period before the test set start are removed.
    embargo_window_months: Duration (in months) to skip between the end of training and start of testing.
    """
    # Convert months to approximate number of daily business observations
    purge_window_days = purge_window_months * 21
    embargo_window_days = embargo_window_months * 21

    tscv = TimeSeriesSplit(n_splits=n_splits)
    purged_scores = []
    
    print(f"\n--- Purged & Embargo Walk-Forward CV (n_splits={n_splits}, Purge: {purge_window_months}M, Embargo: {embargo_window_months}M) ---")

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        # Apply Purge
        if purge_window_days > 0:
            # The test period starts at dates_index[test_idx[0]]
            # We need to remove training indices that are within purge_window_days prior to this date
            test_start_date = dates_index[test_idx[0]]
            # Filter training indices where the date is more than purge_window_days before test_start_date
            valid_train_indices = [idx for idx in train_idx if dates_index[idx] < (test_start_date - pd.Timedelta(days=purge_window_days))]
            train_idx_purged = np.array(valid_train_indices)
        else:
            train_idx_purged = train_idx

        # Apply Embargo
        if embargo_window_days > 0:
            # The training period ends at dates_index[train_idx[-1]]
            # We need to filter test indices that are within embargo_window_days after this date
            train_end_date = dates_index[train_idx[-1]]
            # Filter test indices where the date is more than embargo_window_days after train_end_date
            valid_test_indices = [idx for idx in test_idx if dates_index[idx] > (train_end_date + pd.Timedelta(days=embargo_window_days))]
            test_idx_embargoed = np.array(valid_test_indices)
        else:
            test_idx_embargoed = test_idx
        
        # Ensure there's still enough data for training and testing after purging/embargoing
        if len(train_idx_purged) == 0 or len(test_idx_embargoed) == 0:
            print(f"Fold {fold+1}: Skipped due to insufficient data after purging/embargoing.")
            purged_scores.append(np.nan) # Append NaN for skipped folds
            continue

        X_train, X_test = X.iloc[train_idx_purged], X.iloc[test_idx_embargoed]
        y_train, y_test = y.iloc[train_idx_purged], y.iloc[test_idx_embargoed]
        
        model_fold = RandomForestRegressor(**model_config)
        model_fold.fit(X_train, y_train)
        r2_fold = model_fold.score(X_test, y_test)
        purged_scores.append(r2_fold)

        train_start, train_end = dates_index[train_idx_purged.min()], dates_index[train_idx_purged.max()]
        test_start, test_end = dates_index[test_idx_embargoed.min()], dates_index[test_idx_embargoed.max()]
        print(f"Fold {fold+1} (Purged+Embargo): Train from {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}, Test from {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')} -> R2: {r2_fold:.4f}")

    purged_mean_r2 = np.nanmean(purged_scores) # Use nanmean to handle skipped folds
    purged_std_r2 = np.nanstd(purged_scores)
    
    print(f"\nMean CV R2 (Purged+Embargo): {purged_mean_r2:.4f} +/- {purged_std_r2:.4f}")

    # Compare with standard (non-purged) expanding window CV
    standard_scores = expanding_results['R2'].dropna().values # Get R2 values from previous expanding window CV
    standard_mean_r2 = np.mean(standard_scores)
    
    leakage_inflation = standard_mean_r2 - purged_mean_r2
    
    print(f"\n--- Leakage Analysis ---")
    print(f"Mean CV R2 (Standard Expanding Window): {standard_mean_r2:.4f}")
    print(f"Mean CV R2 (Purged+Embargo): {purged_mean_r2:.4f}")
    print(f"Leakage Inflation (Standard R2 - Purged R2): {leakage_inflation:.4f}")
    if leakage_inflation > 0.02: # Heuristic threshold
        print("\nWarning: Significant leakage inflation detected. The model's apparent performance was likely illusory due to data leakage.")
    elif leakage_inflation > 0:
        print("\nNote: Some leakage inflation detected, indicating potential information leakage.")
    else:
        print("\nGood: Minimal or no leakage inflation detected.")

    return purged_scores, purged_mean_r2, purged_std_r2, leakage_inflation

purged_scores, purged_mean_r2, purged_std_r2, leakage_inflation = purged_embargo_walk_forward_cv(X, y, dates_aligned, model_config, n_splits=5, purge_window_months=12, embargo_window_months=1)
```

### Explanation of Execution

The `Leakage Inflation` metric reveals the difference between the standard walk-forward $R^2$ and the more robust purged and embargoed $R^2$. If this value is positive and significant (e.g., $ > 0.02 $), it strongly suggests that the model's apparent performance was inflated due to information leakage, not genuine predictive power. Sarah takes this number very seriously; it's a direct measure of how much of the model's "alpha" might be an illusion. A high leakage inflation is a major red flag for model deployment, indicating the model is likely overfit to artifacts of the data rather than true underlying signals. This step ensures that her investment recommendations are based on a reasonable and adequate basis, free from critical data validation errors.

---

## 5. Stress-Testing Across Market Regimes: Identifying "Ticking Time Bombs"

### Story + Context + Real-World Relevance

A model might perform well in benign bull markets but collapse during bear markets or periods of high volatility. Sarah needs to understand the model's robustness across different **market regimes** to identify these "ticking time bombs." Relying on average performance metrics (like mean $R^2$) can mask critical weaknesses during periods when accurate predictions are most crucial. For instance, a credit model's performance in a recession is far more important than its performance during an expansion.

Sarah decides to classify historical market regimes based on common financial indicators: S&P 500 returns for "bull" vs. "bear" and VIX levels for "high volatility" vs. "low volatility." She will then re-evaluate the model's performance (specifically, the distribution of residuals) within each regime. A model that works in bull markets but fails in bear markets, or one whose errors explode during high volatility, is unacceptable for Alpha Capital.

The regimes are defined as:
*   **Bull:** 12-month rolling average of S&P 500 returns is positive.
*   **Bear:** 12-month rolling average of S&P 500 returns is negative.
*   **High Volatility:** VIX index is above its median.
*   **Low Volatility:** VIX index is below its median.

These are then combined into four primary regimes: Bull-LowVol, Bull-HighVol, Bear-LowVol, and Bear-HighVol.

```python
# --- Fetch S&P 500 and VIX data for regime classification ---
# Adjust dates to cover the entire simulated data range
start_date_yf = dates_aligned.min().strftime('%Y-%m-%d')
end_date_yf = dates_aligned.max().strftime('%Y-%m-%d')

sp500 = yf.download('^GSPC', start=start_date_yf, end=end_date_yf, interval='1mo')['Adj Close']
vix = yf.download('^VIX', start=start_date_yf, end=end_date_yf, interval='1mo')['Adj Close']

# Calculate S&P 500 returns and rolling mean for bull/bear
sp500_ret = sp500.pct_change().dropna()
regimes_df = pd.DataFrame(index=sp500_ret.index)
regimes_df['bull'] = (sp500_ret.rolling(12).mean() > 0).astype(int)

# Classify VIX for high/low volatility
vix_median = vix.median()
regimes_df['high_vol'] = (vix > vix_median).astype(int)

# Create combined regime labels
def get_regime_label(row):
    if row['bull'] and not row['high_vol']:
        return 'Bull-LowVol'
    elif row['bull'] and row['high_vol']:
        return 'Bull-HighVol'
    elif not row['bull'] and row['high_vol']:
        return 'Bear-HighVol'
    else: # not row['bull'] and not row['high_vol']
        return 'Bear-LowVol'

regimes_df['regime'] = regimes_df.apply(get_regime_label, axis=1)
regimes_df = regimes_df[['regime']]

print("\n--- Market Regimes Distribution ---")
print(regimes_df['regime'].value_counts())

# --- Prepare evaluation DataFrame with actuals, predictions, and regimes ---
# To perform regime-conditional analysis, we need predictions for a continuous period
# We will use the last trained model from the expanding window CV for full data prediction
# or retrain once on the entire dataset for consistency with the spirit of evaluation.
# For simplicity, let's retrain the model on the full data and make predictions.
final_model = RandomForestRegressor(**model_config)
final_model.fit(X, y)
y_pred = pd.Series(final_model.predict(X), index=X.index, name='y_predicted')

eval_df = pd.DataFrame({
    'y_actual': y,
    'y_predicted': y_pred
}, index=dates_aligned)

# Align regimes with eval_df (monthly regimes to daily data)
# Use forward fill to propagate monthly regime labels to daily observations
eval_df = eval_df.resample('D').mean().ffill().bfill() # fill missing daily dates, then forward fill
eval_df['regime'] = regimes_df['regime'].resample('D').ffill().reindex(eval_df.index).ffill().bfill() # Align & fill
eval_df = eval_df.dropna() # Drop any rows where regime info is still missing
eval_df['residual'] = eval_df['y_actual'] - eval_df['y_predicted']

print(f"\nEvaluation DataFrame Head with Regimes:\n{eval_df.head()}")

# --- Regime-Conditional Performance Analysis ---
regime_perf = eval_df.groupby('regime').apply(lambda g: pd.Series({
    'R2': 1 - ( (g['y_actual'] - g['y_predicted'])**2 ).sum() / ( (g['y_actual'] - g['y_actual'].mean())**2 ).sum(),
    'RMSE': np.sqrt(mean_squared_error(g['y_actual'], g['y_predicted'])),
    'n_obs': len(g),
    'Avg_Actual': g['y_actual'].mean(),
    'Avg_Predicted': g['y_predicted'].mean(),
    'Bias': (g['y_predicted'] - g['y_actual']).mean()
})).round(4)

print("\n--- Regime-Conditional Performance Metrics (Full Data) ---")
print(regime_perf)

# --- Visualization: Residual distributions across regimes ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Box plot of residuals by regime
sns.boxplot(x='regime', y='residual', data=eval_df, ax=axes[0], palette='viridis')
axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
axes[0].set_title('Prediction Residuals by Market Regime')
axes[0].set_xlabel('Market Regime')
axes[0].set_ylabel('Residual (Actual - Predicted)')
axes[0].tick_params(axis='x', rotation=30)

# Bar chart of R2 by regime
regime_perf['R2'].plot(kind='bar', ax=axes[1], color='steelblue', edgecolor='black')
axes[1].set_title('Out-of-Sample R2 by Market Regime')
axes[1].set_xlabel('Market Regime')
axes[1].set_ylabel('R2 Score')
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.show()
```

### Explanation of Execution

The `Regime-Conditional Performance Metrics` table and the box plots of residuals are extremely insightful for Sarah. She can now clearly see how the model's $R^2$ changes across different market conditions and observe the distribution of prediction errors. For instance, if the $R^2$ is high in 'Bull-LowVol' but near zero or negative in 'Bear-HighVol', it signals a critical weakness. A wider box plot of residuals (indicating larger errors) or a median shifted significantly from zero in 'Bear-HighVol' would be a major warning. This analysis directly informs Sarah if the model is robust enough for all market environments or if it's a fair-weather performer, helping her prevent capital loss during periods of market stress.

---

## 6. Diagnosing Bias-Variance with Learning Curves

### Story + Context + Real-World Relevance

Sarah understands that a model's performance is affected by its complexity and the amount of training data. To diagnose if the factor model is suffering from **high bias (underfitting)** or **high variance (overfitting)**, and whether acquiring more data would improve its performance, she uses **Learning Curves**. A learning curve plots the training and cross-validation scores as a function of the training set size.

The expected behaviors are:
*   **High bias (underfitting):** Both training and validation scores are low and converge at a low value. More data does not help; the model is too simple to capture the underlying patterns.
*   **High variance (overfitting):** Training score is high, but validation score is low, with a significant gap between them. As training data increases, the gap narrows, and the validation score eventually rises. More data helps.
*   **Optimal complexity:** Training score is moderately high, validation score is close to it, and the gap is small and stable.

For financial models, high variance is a common pathology (too many features relative to samples, or model is too flexible), while high bias can occur with overly restrictive models (e.g., linear models on non-linear data). Sarah wants to understand where this model stands to inform potential next steps: feature engineering, model complexity adjustment, or data acquisition.

```python
def plot_learning_curve(X, y, model_config, cv_strategy, scoring_metric='r2', n_jobs=-1):
    """
    Generates and plots a learning curve.
    cv_strategy: An instance of a cross-validation splitter (e.g., TimeSeriesSplit(n_splits=5)).
    """
    model_lc = RandomForestRegressor(**model_config)
    
    train_sizes, train_scores, test_scores = learning_curve(
        model_lc, X, y, cv=cv_strategy, train_sizes=np.linspace(0.2, 1.0, 8),
        scoring=scoring_metric, n_jobs=n_jobs
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="b")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="r")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label="Training Score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label="Cross-validation Score")

    plt.xlabel('Training Set Size (Observations)')
    plt.ylabel(f'{scoring_metric.upper()} Score')
    plt.title('Learning Curve: Bias-Variance Diagnosis')
    plt.legend(loc="best")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    # Diagnose bias-variance
    final_train_score = train_scores_mean[-1]
    final_test_score = test_scores_mean[-1]
    gap_at_max_data = final_train_score - final_test_score

    diagnosis = {}
    if final_train_score < 0.2 and final_test_score < 0.1: # Low scores for both
        diagnosis['Converged'] = False
        diagnosis['Recommendation'] = "High bias (underfitting). Model is too simple; consider more features or a more complex model."
        diagnosis['Gap_Converged'] = False
    elif gap_at_max_data > 0.15: # Significant gap
        diagnosis['Converged'] = False
        diagnosis['Recommendation'] = "High variance (overfitting). More data might help reduce the gap, or consider regularization/simpler model."
        diagnosis['Gap_Converged'] = False
    elif final_test_score > 0.05 and gap_at_max_data < 0.1: # Decent test score and small gap
        diagnosis['Converged'] = True
        diagnosis['Recommendation'] = "Optimal complexity. Model performs well, and additional data might provide diminishing returns."
        diagnosis['Gap_Converged'] = True
    else: # Catch-all for other scenarios
        diagnosis['Converged'] = False
        diagnosis['Recommendation'] = "Mixed diagnosis. Review scores and gap carefully."
        diagnosis['Gap_Converged'] = False
    
    print(f"\n--- Learning Curve Diagnosis ---")
    print(f"Final Training Score: {final_train_score:.4f}")
    print(f"Final Cross-validation Score: {final_test_score:.4f}")
    print(f"Gap at Max Training Data: {gap_at_max_data:.4f}")
    print(f"Recommendation: {diagnosis['Recommendation']}")
    
    return diagnosis

# Using TimeSeriesSplit for the learning curve to respect temporal order
tscv_lc = TimeSeriesSplit(n_splits=5)
learning_curve_diagnosis = plot_learning_curve(X, y, model_config, tscv_lc, scoring_metric='r2')
```

### Explanation of Execution

The learning curve provides a visual and quantitative diagnosis of the model's bias-variance tradeoff. Sarah can see if the training and cross-validation scores converge and at what level. If the two curves are far apart, it suggests high variance (overfitting), implying the model is too complex for the given data or needs more data. If both curves are low and flat, it indicates high bias (underfitting), meaning the model is too simple. This helps Sarah make informed decisions: if it's high variance, she might consider regularization or increasing the data size; if it's high bias, she might explore more complex features or model architectures. This is a crucial step in understanding the model's fundamental limitations and potential for improvement.

---

## 7. Quantifying Model Stability and Overfitting: The Validation Report

### Story + Context + Real-World Relevance

After running multiple rigorous validation tests, Sarah now needs to synthesize all findings into a concise, actionable "Model Validation Report." This report is the final deliverable for Alpha Capital's Investment Committee and the Model Risk Management (MRM) team. It must clearly communicate the model's strengths, weaknesses, and a go/no-go recommendation based on quantified metrics of stability and overfitting.

Key metrics Sarah needs to present include:

1.  **Overfitting Ratio (OFR):** Measures the relative degradation from in-sample to out-of-sample performance.
    $$ \text{OFR} = \frac{P_{\text{in-sample}} - P_{\text{out-of-sample}}}{P_{\text{in-sample}}} $$
    where $P_{\text{in-sample}}$ is the $R^2$ from a random split (representing potentially inflated in-sample performance) and $P_{\text{out-of-sample}}$ is the $R^2$ from a robust temporal split (e.g., the mean of purged CV or expanding window CV).
    *   OFR $\approx 0$: No overfitting.
    *   OFR $> 0.5$: Significant overfitting, model is essentially memorizing.

2.  **Performance Stability Ratio (PSR):** Measures the consistency of out-of-sample performance across different folds/regimes.
    $$ \text{PSR} = \frac{\min_{k} P^{(k)}_{\text{test}}}{\max_{k} P^{(k)}_{\text{test}}} $$
    where $P^{(k)}_{\text{test}}$ is the test performance (e.g., $R^2$) for fold $k$.
    *   PSR $\approx 1$: Performance is stable.
    *   PSR $< 0$: Model performs well in some periods but has negative performance in others (e.g., negative $R^2$)—a red flag.

Sarah will gather all calculated metrics, regime performance, and the learning curve diagnosis into a structured report template, ready for presentation.

```python
# --- Calculate P_in-sample (random split R2) and P_out-of-sample (purged CV mean R2) ---
# Use the previously calculated r2_random from Section 1 for P_in-sample
# Use purged_mean_r2 from Section 4 for P_out-of-sample (most robust OOS estimate)

P_in_sample = r2_random
P_out_of_sample = purged_mean_r2

# Ensure P_in_sample is not zero to avoid division by zero
if P_in_sample == 0:
    overfitting_ratio = np.nan
else:
    overfitting_ratio = (P_in_sample - P_out_of_sample) / P_in_sample

# --- Performance Stability Ratio (PSR) ---
# We use the R2 scores from the purged_embargo_walk_forward_cv as our fold-wise P_test
# Filter out NaN values if some folds were skipped
valid_purged_scores = [s for s in purged_scores if not np.isnan(s)]

if len(valid_purged_scores) > 1:
    min_r2_fold = np.min(valid_purged_scores)
    max_r2_fold = np.max(valid_purged_scores)
    if max_r2_fold == 0: # Avoid division by zero, especially if all R2s are zero or negative
        performance_stability_ratio = np.nan
    else:
        performance_stability_ratio = min_r2_fold / max_r2_fold
else: # Not enough folds to calculate min/max, or all were NaN
    performance_stability_ratio = np.nan

# --- Worst Regime R2 ---
worst_regime_r2 = regime_perf['R2'].min()
best_regime_r2 = regime_perf['R2'].max()
regime_spread = best_regime_r2 - worst_regime_r2

# --- Learning Curve Gap ---
# Extract from the diagnosis dict
lc_gap_at_max_data = learning_curve_diagnosis['Gap_at_Max_Training_Data']
lc_converged = learning_curve_diagnosis['Converged']
lc_recommendation_text = learning_curve_diagnosis['Recommendation']

# --- Assemble the Model Validation Report ---
report = f"""
================================================================================
MODEL VALIDATION REPORT
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: {model.__class__.__name__}
Asset Universe: Simulated Equity Factor Model
================================================================================

1. IN-SAMPLE PERFORMANCE (Random Split)
---------------------------------------
Training R2 (Random Split):   {r2_random:.4f}
(This value is typically inflated due to information leakage and should be viewed with caution.)

2. OUT-OF-SAMPLE PERFORMANCE (Robust Walk-Forward CV)
------------------------------------------------------
Mean CV R2 (Expanding Window): {expanding_mean_r2:.4f} +/- {expanding_std_r2:.4f}
Mean CV R2 (Sliding Window):   {sliding_mean_r2:.4f} +/- {sliding_std_r2:.4f}
Mean CV R2 (Purged+Embargo):   {purged_mean_r2:.4f} +/- {purged_std_r2:.4f}

3. OVERFITTING DIAGNOSTICS
--------------------------
Overfitting Gap (Random R2 - Temporal R2): {overfitting_gap:.4f}
Leakage Inflation (Standard R2 - Purged R2): {leakage_inflation:.4f}
Overfitting Ratio (OFR): {overfitting_ratio:.4f}
    {'🟢 No significant overfitting detected' if overfitting_ratio < 0.3 else '🟡 Moderate overfitting' if overfitting_ratio < 0.5 else '🔴 High overfitting'}

Performance Stability Ratio (PSR): {performance_stability_ratio:.4f}
    {'🟢 Stable performance across folds' if performance_stability_ratio > 0.5 else '🟡 Volatile performance across folds' if performance_stability_ratio >= 0 else '🔴 Negative performance in some folds - major red flag'}

4. REGIME STABILITY
-------------------
{regime_perf.to_string()}

Worst-Regime R2 (min across regimes): {worst_regime_r2:.4f}
Regime Spread (Best R2 - Worst R2): {regime_spread:.4f}
    {'🟢 Stable across regimes' if regime_spread < 0.2 else '🟡 Performance varies across regimes' if regime_spread < 0.4 else '🔴 Unstable performance across regimes - ticking time bomb'}

5. LEARNING CURVE DIAGNOSIS
---------------------------
Converged: {'Yes' if lc_converged else 'No'}
Gap at Max Training Data (Train R2 - CV R2): {lc_gap_at_max_data:.4f}
Recommendation: {lc_recommendation_text}

6. OVERALL RECOMMENDATION
-------------------------
[ ] Approved for production
[ ] Requires further development (e.g., more data, feature engineering, model complexity adjustment, further stress testing)
[ ] Rejected (Model deemed unreliable for deployment)

Based on the quantitative metrics and analysis:
- The model exhibits {'low' if overfitting_ratio < 0.3 else 'moderate' if overfitting_ratio < 0.5 else 'high'} overfitting.
- Performance stability across validation folds is {'good' if performance_stability_ratio > 0.5 else 'concerning' if performance_stability_ratio >= 0 else 'poor'}.
- Regime analysis indicates {'robust' if regime_spread < 0.2 else 'variable' if regime_spread < 0.4 else 'unreliable'} performance across different market conditions.
- The learning curve suggests: {lc_recommendation_text}.

Sarah will provide her final "Go/No-Go" decision and detailed rationale based on these findings.
"""

print(report)
```

### Explanation of Execution

This comprehensive `Model Validation Report` provides Sarah with a holistic view of the model's reliability. Each metric (Overfitting Ratio, Performance Stability Ratio, Leakage Inflation, Regime Spread) is explicitly calculated and interpreted. This allows her to quantify the risks associated with the model's deployment. The report serves as the basis for a data-driven investment decision, fulfilling her ethical obligations as a CFA Charterholder and aligning with Alpha Capital's rigorous risk management framework. For instance, if the Overfitting Ratio is high and the Performance Stability Ratio is low, coupled with poor performance in 'Bear-HighVol' regimes, Sarah will likely recommend against deploying the model without significant further development or rejection. This is how quantitative validation directly translates into prudent capital allocation and risk management in a financial institution.
