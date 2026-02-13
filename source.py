import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit, learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import datetime as dt
import warnings

# Suppress warnings for cleaner output in an application context
warnings.filterwarnings('ignore')

# Set plotting styles globally for consistency in the module's plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100


def generate_synthetic_financial_data(n_samples: int = 252 * 10, start_date: str = '2010-01-01', seed: int = 42) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    """
    Generates a synthetic financial time-series dataset for demonstration purposes.
    Features (X) are slightly lagged and correlated with the target (y).

    Args:
        n_samples (int): Number of daily samples to generate (e.g., 252 * 10 for 10 years).
        start_date (str): Start date for the time series.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]: X (features), y (target), aligned dates.
    """
    np.random.seed(seed)

    dates = pd.date_range(start=start_date, periods=n_samples, freq='B')

    # Simulate some financial factors (e.g., momentum, value)
    factors = pd.DataFrame(
        np.random.randn(n_samples, 5),
        index=dates,
        columns=[f'Factor_{i}' for i in range(1, 6)]
    )
    factors = factors.cumsum()  # Make them more "time-series like"

    # Simulate an underlying "true" process for target variable
    true_signal = (
        0.5 * factors['Factor_1'].shift(1)
        + 0.3 * factors['Factor_2'].shift(2)
        - 0.2 * factors['Factor_3'].shift(1)
    )

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

    return X, y, dates_aligned


def evaluate_split_strategy(X: pd.DataFrame, y: pd.Series, model_config: dict) -> tuple[float, float, float]:
    """
    Compares random vs. temporal splitting for a given model configuration.

    Args:
        X (pd.DataFrame): Feature DataFrame.
        y (pd.Series): Target Series.
        model_config (dict): Dictionary of model parameters for RandomForestRegressor.

    Returns:
        tuple[float, float, float]: R2 with random split, R2 with temporal split, overfitting gap.
    """
    model_random = RandomForestRegressor(**model_config)
    model_temporal = RandomForestRegressor(**model_config)

    # 1) Random Split
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y, test_size=0.25, random_state=model_config.get('random_state', 42)
    )
    model_random.fit(X_train_r, y_train_r)
    r2_random = model_random.score(X_test_r, y_test_r)

    # 2) Temporal Split
    split_idx = int(len(X) * 0.75)
    X_train_t, X_test_t = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_t, y_test_t = y.iloc[:split_idx], y.iloc[split_idx:]

    model_temporal.fit(X_train_t, y_train_t)
    r2_temporal = model_temporal.score(X_test_t, y_test_t)

    print("\n--- Model Performance Comparison (Random vs. Temporal Split) ---")
    print(f"R2 with RANDOM split:   {r2_random:.4f}")
    print(f"R2 with TEMPORAL split: {r2_temporal:.4f}")

    overfitting_gap = r2_random - r2_temporal
    print(f"Overfitting Gap (Random R2 - Temporal R2): {overfitting_gap:.4f}")

    if overfitting_gap > 0.05:  # heuristic threshold
        print(
            "\nWarning: Significant overfitting gap detected! "
            "Random splitting artificially inflated performance."
        )

    return r2_random, r2_temporal, overfitting_gap


def expanding_window_cv(X: pd.DataFrame, y: pd.Series, dates_index: pd.DatetimeIndex, model_config: dict, n_splits: int = 5) -> tuple[pd.DataFrame, float, float]:
    """
    Performs expanding window walk-forward cross-validation.

    Args:
        X (pd.DataFrame): Feature DataFrame.
        y (pd.Series): Target Series.
        dates_index (pd.DatetimeIndex): Datetime index aligned with X and y.
        model_config (dict): Dictionary of model parameters for RandomForestRegressor.
        n_splits (int): Number of splits for TimeSeriesSplit.

    Returns:
        tuple[pd.DataFrame, float, float]: DataFrame of fold results, mean R2, std R2.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []

    print(
        f"\n--- Expanding Window Walk-Forward Cross-Validation (n_splits={n_splits}) ---"
    )

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        train_start, train_end = dates_index[train_idx.min()], dates_index[train_idx.max()]
        test_start, test_end = dates_index[test_idx.min()], dates_index[test_idx.max()]

        model_fold = RandomForestRegressor(**model_config)
        model_fold.fit(X_train, y_train)

        r2_fold = model_fold.score(X_test, y_test)

        fold_results.append(
            {
                'Fold': fold + 1,
                'Train_Start': train_start.strftime('%Y-%m-%d'),
                'Train_End': train_end.strftime('%Y-%m-%d'),
                'Test_Start': test_start.strftime('%Y-%m-%d'),
                'Test_End': test_end.strftime('%Y-%m-%d'),
                'R2': r2_fold,
                'N_Train': len(train_idx),
                'N_Test': len(test_idx),
            }
        )

        print(
            f"Fold {fold+1}: Train from {train_start.strftime('%Y-%m-%d')} "
            f"to {train_end.strftime('%Y-%m-%d')}, Test from {test_start.strftime('%Y-%m-%d')} "
            f"to {test_end.strftime('%Y-%m-%d')} -> R2: {r2_fold:.4f}"
        )

    results_df = pd.DataFrame(fold_results)
    mean_r2 = results_df['R2'].mean()
    std_r2 = results_df['R2'].std()

    print(f"\nMean CV R2 (Expanding Window): {mean_r2:.4f} +/- {std_r2:.4f}")

    return results_df, mean_r2, std_r2


def sliding_window_cv(
    X: pd.DataFrame,
    y: pd.Series,
    dates_index: pd.DatetimeIndex,
    model_config: dict,
    train_window_size_months: int = 60,
    test_window_size_months: int = 12,
) -> tuple[pd.DataFrame, float, float]:
    """
    Performs sliding window walk-forward cross-validation.

    Args:
        X (pd.DataFrame): Feature DataFrame.
        y (pd.Series): Target Series.
        dates_index (pd.DatetimeIndex): Datetime index aligned with X and y.
        model_config (dict): Dictionary of model parameters for RandomForestRegressor.
        train_window_size_months (int): Fixed number of months for training.
        test_window_size_months (int): Fixed number of months for testing (step size).

    Returns:
        tuple[pd.DataFrame, float, float]: DataFrame of sliding window results, mean R2, std R2.
    """
    # Convert months to approximate business days (21 business days/month)
    train_window_size = train_window_size_months * 21
    test_window_size = test_window_size_months * 21

    sliding_results = []

    print(
        f"\n--- Sliding Window Walk-Forward Cross-Validation "
        f"(Train Window: {train_window_size_months} months, "
        f"Test Window: {test_window_size_months} months) ---"
    )

    # Iterate through the data to create sliding windows
    for i in range(train_window_size, len(X), test_window_size):
        train_start_idx = i - train_window_size
        train_end_idx = i  # End of training window is start of test window
        test_start_idx = i
        test_end_idx = min(i + test_window_size, len(X))  # Ensure test window doesn't go OOB

        if test_start_idx >= test_end_idx:  # No more full test windows
            break

        X_train = X.iloc[train_start_idx:train_end_idx]
        X_test = X.iloc[test_start_idx:test_end_idx]
        y_train = y.iloc[train_start_idx:train_end_idx]
        y_test = y.iloc[test_start_idx:test_end_idx]

        if len(y_train) == 0 or len(y_test) == 0:
            print(f"Skipping fold starting at {dates_index[test_start_idx].strftime('%Y-%m-%d')} due to insufficient data (Train: {len(y_train)}, Test: {len(y_test)})")
            continue

        model_fold = RandomForestRegressor(**model_config)
        model_fold.fit(X_train, y_train)
        r2_fold = model_fold.score(X_test, y_test)

        sliding_results.append(
            {
                'Test_Period_Start': dates_index[test_start_idx],
                'Test_Period_End': dates_index[test_end_idx - 1],
                'R2': r2_fold,
                'N_Train': len(X_train),
                'N_Test': len(X_test),
            }
        )

        print(
            f"Test Period: {dates_index[test_start_idx].strftime('%Y-%m-%d')} "
            f"to {dates_index[test_end_idx-1].strftime('%Y-%m-%d')} -> R2: {r2_fold:.4f}"
        )

    sliding_df = pd.DataFrame(sliding_results)

    if not sliding_df.empty:
        mean_r2_sliding = sliding_df['R2'].mean()
        std_r2_sliding = sliding_df['R2'].std()
        print(f"\nMean CV R2 (Sliding Window): {mean_r2_sliding:.4f} +/- {std_r2_sliding:.4f}")
    else:
        mean_r2_sliding = np.nan
        std_r2_sliding = np.nan
        print("\nNo sliding window results generated.")

    return sliding_df, mean_r2_sliding, std_r2_sliding


def plot_sliding_window_r2(sliding_df: pd.DataFrame):
    """
    Plots the R2 performance over time for sliding window cross-validation.

    Args:
        sliding_df (pd.DataFrame): DataFrame containing sliding window results.
    """
    if sliding_df.empty:
        print("No data to plot for sliding window R2.")
        return

    plt.figure(figsize=(14, 7))
    plt.plot(
        sliding_df['Test_Period_Start'],
        sliding_df['R2'],
        'b-o',
        markersize=4,
        label='Out-of-Sample R2',
    )
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Zero R2 Baseline')
    plt.xlabel('Test Period Start')
    plt.ylabel('Out-of-Sample R2')
    plt.title('Sliding-Window Performance Over Time')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def purged_embargo_walk_forward_cv(
    X: pd.DataFrame,
    y: pd.Series,
    dates_index: pd.DatetimeIndex,
    model_config: dict,
    n_splits: int = 5,
    purge_window_months: int = 12,
    embargo_window_months: int = 1,
    expanding_results_df: pd.DataFrame = None  # Added for comparison
) -> tuple[list, float, float, float]:
    """
    Performs walk-forward cross-validation with purging and embargoing.

    Args:
        X (pd.DataFrame): Feature DataFrame.
        y (pd.Series): Target Series.
        dates_index (pd.DatetimeIndex): Datetime index aligned with X and y.
        model_config (dict): Dictionary of model parameters for RandomForestRegressor.
        n_splits (int): Number of splits for TimeSeriesSplit.
        purge_window_months (int): Duration (months) of forward label horizon to purge.
        embargo_window_months (int): Duration (months) to skip between train end and test start.
        expanding_results_df (pd.DataFrame): Results from standard expanding window CV for leakage comparison.

    Returns:
        tuple[list, float, float, float]: List of purged scores, mean R2, std R2, leakage inflation.
    """
    purge_window_days = purge_window_months * 21  # Approximate business days
    embargo_window_days = embargo_window_months * 21  # Approximate business days

    tscv = TimeSeriesSplit(n_splits=n_splits)
    purged_scores = []

    print(
        f"\n--- Purged & Embargo Walk-Forward CV "
        f"(n_splits={n_splits}, Purge: {purge_window_months}M, Embargo: {embargo_window_months}M) ---"
    )

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        train_idx_purged = np.array(train_idx)
        test_idx_embargoed = np.array(test_idx)

        # --- Purge ---
        if purge_window_days > 0 and len(test_idx) > 0:
            test_start_date = dates_index[test_idx[0]]
            valid_train_indices = [
                idx for idx in train_idx
                if dates_index[idx] < (test_start_date - pd.Timedelta(days=purge_window_days))
            ]
            train_idx_purged = np.array(valid_train_indices)

        # --- Embargo ---
        if embargo_window_days > 0 and len(train_idx) > 0:
            train_end_date = dates_index[train_idx[-1]]
            valid_test_indices = [
                idx for idx in test_idx
                if dates_index[idx] > (train_end_date + pd.Timedelta(days=embargo_window_days))
            ]
            test_idx_embargoed = np.array(valid_test_indices)

        if len(train_idx_purged) == 0 or len(test_idx_embargoed) == 0:
            print(f"Fold {fold+1}: Skipped due to insufficient data after purging/embargoing (Train: {len(train_idx_purged)}, Test: {len(test_idx_embargoed)}).")
            purged_scores.append(np.nan)
            continue

        X_train, X_test = X.iloc[train_idx_purged], X.iloc[test_idx_embargoed]
        y_train, y_test = y.iloc[train_idx_purged], y.iloc[test_idx_embargoed]

        model_fold = RandomForestRegressor(**model_config)
        model_fold.fit(X_train, y_train)
        r2_fold = model_fold.score(X_test, y_test)
        purged_scores.append(r2_fold)

        train_start = dates_index[train_idx_purged.min()]
        train_end = dates_index[train_idx_purged.max()]
        test_start = dates_index[test_idx_embargoed.min()]
        test_end = dates_index[test_idx_embargoed.max()]

        print(
            f"Fold {fold+1} (Purged+Embargo): Train from {train_start.strftime('%Y-%m-%d')} "
            f"to {train_end.strftime('%Y-%m-%d')}, Test from {test_start.strftime('%Y-%m-%d')} "
            f"to {test_end.strftime('%Y-%m-%d')} -> R2: {r2_fold:.4f}"
        )

    purged_mean_r2 = np.nanmean(purged_scores)
    purged_std_r2 = np.nanstd(purged_scores)

    print(f"\nMean CV R2 (Purged+Embargo): {purged_mean_r2:.4f} +/- {purged_std_r2:.4f}")

    leakage_inflation = np.nan
    if expanding_results_df is not None and not expanding_results_df.empty:
        standard_scores = expanding_results_df['R2'].dropna().values
        standard_mean_r2 = np.mean(standard_scores)
        leakage_inflation = standard_mean_r2 - purged_mean_r2

        print("\n--- Leakage Analysis ---")
        print(f"Mean CV R2 (Standard Expanding Window): {standard_mean_r2:.4f}")
        print(f"Mean CV R2 (Purged+Embargo):            {purged_mean_r2:.4f}")
        print(f"Leakage Inflation (Standard R2 - Purged R2): {leakage_inflation:.4f}")

        if leakage_inflation > 0.02:
            print(
                "\nWarning: Significant leakage inflation detected. "
                "Apparent performance was likely inflated due to data leakage."
            )
        elif leakage_inflation > 0:
            print("\nNote: Some leakage inflation detected, indicating potential information leakage.")
        else:
            print("\nGood: Minimal or no leakage inflation detected.")
    else:
        print("\nSkipping leakage analysis: No standard expanding window results provided.")

    return purged_scores, purged_mean_r2, purged_std_r2, leakage_inflation


def classify_market_regimes(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches S&P 500 and VIX data and classifies market regimes.

    Args:
        start_date (str): Start date for data fetching.
        end_date (str): End date for data fetching.

    Returns:
        pd.DataFrame: DataFrame with market regimes ('Bull-LowVol', 'Bull-HighVol', etc.).
    """
    sp500 = yf.download(
        '^GSPC',
        start=start_date,
        end=end_date,
        interval='1mo',
        progress=False
    )['Close']

    vix = yf.download(
        '^VIX',
        start=start_date,
        end=end_date,
        interval='1mo',
        progress=False
    )['Close']

    # Calculate S&P 500 returns and rolling mean for bull/bear
    sp500_ret = sp500.pct_change().dropna()

    regimes_df = pd.DataFrame(index=sp500_ret.index)
    regimes_df['bull'] = (sp500_ret.rolling(12).mean() > 0).astype(int)

    # Classify VIX for high/low volatility
    vix_median = vix.median()
    regimes_df['high_vol'] = (vix > vix_median).astype(int)

    def get_regime_label(row):
        if row['bull'] and not row['high_vol']:
            return 'Bull-LowVol'
        elif row['bull'] and row['high_vol']:
            return 'Bull-HighVol'
        elif (not row['bull']) and row['high_vol']:
            return 'Bear-HighVol'
        else:  # (not row['bull']) and (not row['high_vol'])
            return 'Bear-LowVol'

    regimes_df['regime'] = regimes_df.apply(get_regime_label, axis=1)
    regimes_df = regimes_df[['regime']]

    print("\n--- Market Regimes Distribution ---")
    print(regimes_df['regime'].value_counts())

    return regimes_df


def prepare_evaluation_data(
    X: pd.DataFrame, y: pd.Series, dates_aligned: pd.DatetimeIndex,
    model_config: dict, regimes_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Trains a final model, makes predictions, and aligns with market regimes for evaluation.

    Args:
        X (pd.DataFrame): Feature DataFrame.
        y (pd.Series): Target Series.
        dates_aligned (pd.DatetimeIndex): Datetime index aligned with X and y.
        model_config (dict): Dictionary of model parameters for the final model.
        regimes_df (pd.DataFrame): DataFrame containing market regimes (monthly).

    Returns:
        pd.DataFrame: Evaluation DataFrame with actuals, predictions, residuals, and regimes.
    """
    final_model = RandomForestRegressor(**model_config)
    final_model.fit(X, y)

    y_pred = pd.Series(final_model.predict(X), index=X.index, name='y_predicted')

    eval_df = pd.DataFrame(
        {
            'y_actual': y,
            'y_predicted': y_pred,
        },
        index=dates_aligned,
    )

    # Align regimes with eval_df (monthly regimes to daily data)
    daily_regimes = regimes_df['regime'].resample('D').ffill()
    eval_df['regime'] = daily_regimes.reindex(eval_df.index, method='ffill')
    eval_df = eval_df.dropna()  # Drop any rows where regime couldn't be filled, or if y_actual/y_predicted have NaNs.

    eval_df['residual'] = eval_df['y_actual'] - eval_df['y_predicted']

    print(f"\nEvaluation DataFrame Head with Regimes:\n{eval_df.head()}")

    return eval_df


def calculate_regime_performance(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates performance metrics grouped by market regime.

    Args:
        eval_df (pd.DataFrame): Evaluation DataFrame with actuals, predictions, residuals, and regimes.

    Returns:
        pd.DataFrame: DataFrame of performance metrics per regime.
    """
    regime_perf = (
        eval_df.groupby('regime')
        .apply(
            lambda g: pd.Series(
                {
                    'R2': r2_score(g['y_actual'], g['y_predicted']),  # Using sklearn's r2_score
                    'RMSE': np.sqrt(mean_squared_error(g['y_actual'], g['y_predicted'])),
                    'n_obs': len(g),
                    'Avg_Actual': g['y_actual'].mean(),
                    'Avg_Predicted': g['y_predicted'].mean(),
                    'Bias': (g['y_predicted'] - g['y_actual']).mean(),
                }
            )
        )
        .round(4)
    )

    print("\n--- Regime-Conditional Performance Metrics (Full Data) ---")
    print(regime_perf)

    return regime_perf


def plot_regime_performance(eval_df: pd.DataFrame, regime_perf: pd.DataFrame):
    """
    Visualizes prediction residuals and R2 scores across market regimes.

    Args:
        eval_df (pd.DataFrame): Evaluation DataFrame with residuals and regimes.
        regime_perf (pd.DataFrame): DataFrame of performance metrics per regime.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.boxplot(
        x='regime',
        y='residual',
        data=eval_df,
        ax=axes[0],
        palette='viridis',
    )
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0].set_title('Prediction Residuals by Market Regime')
    axes[0].set_xlabel('Market Regime')
    axes[0].set_ylabel('Residual (Actual - Predicted)')
    axes[0].tick_params(axis='x', rotation=30)

    regime_perf['R2'].plot(kind='bar', ax=axes[1], color='steelblue', edgecolor='black')
    axes[1].set_title('Out-of-Sample R2 by Market Regime')
    axes[1].set_xlabel('Market Regime')
    axes[1].set_ylabel('R2 Score')
    axes[1].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.show()


def plot_learning_curve(X: pd.DataFrame, y: pd.Series, model_config: dict, cv_strategy, scoring_metric: str = 'r2', n_jobs: int = -1) -> dict:
    """
    Generates and plots a learning curve for bias-variance diagnosis.

    Args:
        X (pd.DataFrame): Feature DataFrame.
        y (pd.Series): Target Series.
        model_config (dict): Dictionary of model parameters for RandomForestRegressor.
        cv_strategy: Cross-validation splitter (e.g., TimeSeriesSplit).
        scoring_metric (str): Metric to use for scoring (e.g., 'r2', 'neg_mean_squared_error').
        n_jobs (int): Number of jobs to run in parallel.

    Returns:
        dict: A dictionary containing learning curve diagnosis.
    """
    model_lc = RandomForestRegressor(**model_config)

    train_sizes, train_scores, test_scores = learning_curve(
        model_lc,
        X,
        y,
        cv=cv_strategy,
        train_sizes=np.linspace(0.2, 1.0, 8),
        scoring=scoring_metric,
        n_jobs=n_jobs,
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))

    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color='b',
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color='r',
    )

    plt.plot(train_sizes, train_scores_mean, 'o-', color='b', label='Training Score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='r', label='Cross-validation Score')

    plt.xlabel('Training Set Size (Observations)')
    plt.ylabel(f'{scoring_metric.upper()} Score')
    plt.title('Learning Curve: Bias-Variance Diagnosis')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Diagnose bias-variance
    final_train_score = train_scores_mean[-1]
    final_test_score = test_scores_mean[-1]
    gap_at_max_data = final_train_score - final_test_score

    diagnosis = {}

    # Define thresholds
    HIGH_BIAS_THRESHOLD = 0.1  # If test score is below this, likely high bias
    HIGH_VARIANCE_GAP_THRESHOLD = 0.15  # If gap is above this, likely high variance
    OPTIMAL_TEST_SCORE_THRESHOLD = 0.05  # Minimum acceptable test score for 'optimal'
    OPTIMAL_GAP_THRESHOLD = 0.1  # Maximum acceptable gap for 'optimal'

    if final_test_score < HIGH_BIAS_THRESHOLD and final_train_score < HIGH_BIAS_THRESHOLD + 0.1:
        diagnosis['Converged'] = False
        diagnosis['Recommendation'] = (
            'High bias (underfitting). Model is too simple or features are not predictive; '
            'consider more features, feature engineering, or a more complex model (e.g., deeper trees).'
        )
        diagnosis['Gap_Converged'] = False
    elif gap_at_max_data > HIGH_VARIANCE_GAP_THRESHOLD:
        diagnosis['Converged'] = False
        diagnosis['Recommendation'] = (
            'High variance (overfitting). The model is too complex for the available data; '
            'consider getting more data, simplifying the model (e.g., shallower trees, fewer features), '
            'or increasing regularization.'
        )
        diagnosis['Gap_Converged'] = False
    elif final_test_score >= OPTIMAL_TEST_SCORE_THRESHOLD and gap_at_max_data < OPTIMAL_GAP_THRESHOLD:
        diagnosis['Converged'] = True
        diagnosis['Recommendation'] = (
            'Optimal complexity. The model performs well without severe bias or variance issues; '
            'additional data may provide diminishing returns. Consider fine-tuning hyperparameters.'
        )
        diagnosis['Gap_Converged'] = True
    else:
        diagnosis['Converged'] = False
        diagnosis['Recommendation'] = (
            'Mixed diagnosis. Review scores and gap carefully. '
            'Could be slight bias, slight variance, or overall low predictive power.'
        )
        diagnosis['Gap_Converged'] = False

    diagnosis['Gap_at_Max_Training_Data'] = gap_at_max_data
    diagnosis['Final_Training_Score'] = final_train_score
    diagnosis['Final_CV_Score'] = final_test_score

    print("\n--- Learning Curve Diagnosis ---")
    print(f"Final Training Score: {final_train_score:.4f}")
    print(f"Final Cross-validation Score: {final_test_score:.4f}")
    print(f"Gap at Max Training Data: {gap_at_max_data:.4f}")
    print(f"Recommendation: {diagnosis['Recommendation']}")

    return diagnosis


def generate_validation_report(
    model_name: str,
    r2_random: float, r2_temporal: float, overfitting_gap: float,
    expanding_mean_r2: float, expanding_std_r2: float,
    sliding_mean_r2: float, sliding_std_r2: float,
    purged_scores: list, purged_mean_r2: float, purged_std_r2: float, leakage_inflation: float,
    regime_perf: pd.DataFrame,
    learning_curve_diagnosis: dict
) -> str:
    """
    Assembles a comprehensive model validation report.

    Args:
        model_name (str): Name of the model (e.g., 'RandomForestRegressor').
        r2_random (float): R2 from random train-test split.
        r2_temporal (float): R2 from temporal train-test split.
        overfitting_gap (float): r2_random - r2_temporal.
        expanding_mean_r2 (float): Mean R2 from expanding window CV.
        expanding_std_r2 (float): Standard deviation of R2 from expanding window CV.
        sliding_mean_r2 (float): Mean R2 from sliding window CV.
        sliding_std_r2 (float): Standard deviation of R2 from sliding window CV.
        purged_scores (list): List of R2 scores from purged+embargo CV.
        purged_mean_r2 (float): Mean R2 from purged+embargo CV.
        purged_std_r2 (float): Standard deviation of R2 from purged+embargo CV.
        leakage_inflation (float): Standard R2 - Purged R2.
        regime_perf (pd.DataFrame): DataFrame of performance metrics per regime.
        learning_curve_diagnosis (dict): Dictionary containing learning curve diagnosis.

    Returns:
        str: The formatted validation report.
    """
    P_in_sample = r2_random
    P_out_of_sample = purged_mean_r2

    overfitting_ratio = (P_in_sample - P_out_of_sample) / P_in_sample if P_in_sample != 0 else np.nan

    valid_purged_scores = [s for s in purged_scores if not np.isnan(s)]
    if len(valid_purged_scores) > 1:
        min_r2_fold = np.min(valid_purged_scores)
        max_r2_fold = np.max(valid_purged_scores)
        performance_stability_ratio = min_r2_fold / max_r2_fold if max_r2_fold != 0 else np.nan
    else:
        performance_stability_ratio = np.nan

    worst_regime_r2 = regime_perf['R2'].min()
    best_regime_r2 = regime_perf['R2'].max()
    regime_spread = best_regime_r2 - worst_regime_r2

    lc_gap_at_max_data = learning_curve_diagnosis.get('Gap_at_Max_Training_Data', np.nan)
    lc_converged = learning_curve_diagnosis.get('Converged', False)
    lc_recommendation_text = learning_curve_diagnosis.get('Recommendation', 'N/A')

    report = f"""================================================================================
MODEL VALIDATION REPORT
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: {model_name}
Asset Universe: Simulated Equity Factor Model
================================================================================

1. IN-SAMPLE PERFORMANCE (Random Split)
---------------------------------------
Training R2 (Random Split): {r2_random:.4f}
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
Performance Stability Ratio (PSR): {performance_stability_ratio:.4f}

4. REGIME STABILITY
-------------------
{regime_perf.to_string()}
Worst-Regime R2 (min across regimes): {worst_regime_r2:.4f}
Regime Spread (Best R2 - Worst R2): {regime_spread:.4f}

5. LEARNING CURVE DIAGNOSIS
---------------------------
Converged: {'Yes' if lc_converged else 'No'}
Gap at Max Training Data (Train R2 - CV R2): {lc_gap_at_max_data:.4f}
Recommendation: {lc_recommendation_text}

6. OVERALL RECOMMENDATION
-------------------------
[] Approved for production
[] Requires further development
[] Rejected (Model deemed unreliable for deployment)

Based on the quantitative metrics and analysis:
- The model exhibits {'low' if (not np.isnan(overfitting_ratio) and overfitting_ratio < 0.3) else 'moderate' if (not np.isnan(overfitting_ratio) and overfitting_ratio < 0.5) else 'high' if not np.isnan(overfitting_ratio) else 'N/A'} overfitting.
- Performance stability across validation folds is {'good' if (not np.isnan(performance_stability_ratio) and performance_stability_ratio > 0.5) else 'concerning' if (not np.isnan(performance_stability_ratio) and performance_stability_ratio >= 0) else 'poor' if not np.isnan(performance_stability_ratio) else 'N/A'}.
- Regime analysis indicates {'robust' if (not np.isnan(regime_spread) and regime_spread < 0.2) else 'variable' if (not np.isnan(regime_spread) and regime_spread < 0.4) else 'unreliable' if not np.isnan(regime_spread) else 'N/A'} performance across market conditions.
- The learning curve suggests: {lc_recommendation_text}.

Sarah will provide her final Go/No-Go decision and detailed rationale based on these findings.
"""
    return report


def run_validation_pipeline(
    n_samples: int = 252 * 10,
    data_start_date: str = '2010-01-01',
    model_params: dict = None,
    cv_n_splits: int = 5,
    sliding_train_months: int = 60,
    sliding_test_months: int = 12,
    purge_months: int = 12,
    embargo_months: int = 1
) -> str:
    """
    Orchestrates the entire model validation pipeline from data generation to report generation.

    Args:
        n_samples (int): Number of daily samples for synthetic data.
        data_start_date (str): Start date for synthetic data and market regime data.
        model_params (dict): Dictionary of RandomForestRegressor parameters.
                             Defaults to {'n_estimators': 100, 'max_depth': 5, 'random_state': 42}.
        cv_n_splits (int): Number of splits for TimeSeriesSplit CV.
        sliding_train_months (int): Training window size in months for sliding CV.
        sliding_test_months (int): Test window size in months for sliding CV.
        purge_months (int): Purge window in months for purged+embargo CV.
        embargo_months (int): Embargo window in months for purged+embargo CV.

    Returns:
        str: The final model validation report.
    """
    if model_params is None:
        model_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'random_state': 42,
        }

    # 1. Data Generation
    X, y, dates_aligned = generate_synthetic_financial_data(n_samples=n_samples, start_date=data_start_date)

    # 2. Split Strategy Evaluation
    r2_random, r2_temporal, overfitting_gap = evaluate_split_strategy(X, y, model_params)

    # 3. Expanding Window CV
    expanding_results, expanding_mean_r2, expanding_std_r2 = expanding_window_cv(
        X, y, dates_aligned, model_params, n_splits=cv_n_splits
    )

    # 4. Sliding Window CV
    sliding_results_df, sliding_mean_r2, sliding_std_r2 = sliding_window_cv(
        X, y, dates_aligned, model_params,
        train_window_size_months=sliding_train_months,
        test_window_size_months=sliding_test_months,
    )
    plot_sliding_window_r2(sliding_results_df)

    # 5. Purged & Embargo Walk-Forward CV
    purged_scores, purged_mean_r2, purged_std_r2, leakage_inflation = purged_embargo_walk_forward_cv(
        X, y, dates_aligned, model_params, n_splits=cv_n_splits,
        purge_window_months=purge_months, embargo_window_months=embargo_months,
        expanding_results_df=expanding_results  # Pass for leakage analysis
    )

    # 6. Market Regime Classification
    end_date_yf = dates_aligned.max().strftime('%Y-%m-%d')
    start_date_yf = dates_aligned.min().strftime('%Y-%m-%d')
    regimes_df = classify_market_regimes(start_date_yf, end_date_yf)

    # 7. Prepare Evaluation Data
    eval_df = prepare_evaluation_data(X, y, dates_aligned, model_params, regimes_df)

    # 8. Regime-Conditional Performance Analysis
    regime_perf = calculate_regime_performance(eval_df)
    plot_regime_performance(eval_df, regime_perf)

    # 9. Learning Curve Diagnosis
    tscv_lc = TimeSeriesSplit(n_splits=cv_n_splits)
    learning_curve_diagnosis = plot_learning_curve(X, y, model_params, tscv_lc, scoring_metric='r2')

    # 10. Generate Report
    report = generate_validation_report(
        model_name="RandomForestRegressor",
        r2_random=r2_random, r2_temporal=r2_temporal, overfitting_gap=overfitting_gap,
        expanding_mean_r2=expanding_mean_r2, expanding_std_r2=expanding_std_r2,
        sliding_mean_r2=sliding_mean_r2, sliding_std_r2=sliding_std_r2,
        purged_scores=purged_scores, purged_mean_r2=purged_mean_r2, purged_std_r2=purged_std_r2,
        leakage_inflation=leakage_inflation,
        regime_perf=regime_perf,
        learning_curve_diagnosis=learning_curve_diagnosis
    )

    print(report)
    return report


if __name__ == '__main__':
    # Example of how to run the pipeline with custom parameters
    report_output = run_validation_pipeline(
        n_samples=252 * 5,  # Use 5 years of data for a quicker demo
        data_start_date='2015-01-01',
        model_params={
            'n_estimators': 50,
            'max_depth': 4,
            'random_state': 42,
        },
        cv_n_splits=3,  # Fewer splits for quicker demo
        sliding_train_months=36,
        sliding_test_months=6,
        purge_months=6,
        embargo_months=1
    )
    # The report string is printed by the function and also returned.
    # In an app.py, you might display this report in a UI or save it.
    # For example:
    # with open("validation_report.txt", "w") as f:
    #     f.write(report_output)
