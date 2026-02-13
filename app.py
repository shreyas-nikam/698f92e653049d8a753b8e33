

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from source import (
    generate_synthetic_financial_data,
    evaluate_split_strategy,
    expanding_window_cv,
    sliding_window_cv,
    purged_embargo_walk_forward_cv,
    calculate_regime_performance, # Fixed: Changed from analyze_regime_performance to calculate_regime_performance as per ImportError
    plot_learning_curve,
    generate_model_validation_report
)
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import TimeSeriesSplit

st.set_page_config(page_title="QuLab: Lab 6: Model Validation", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: Lab 6: Model Validation")
st.divider()

# Initialize session state variables
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
    st.session_state.X = None
    st.session_state.y = None
    st.session_state.dates_aligned = None
    st.session_state.model_config = None
    st.session_state.r2_random = None
    st.session_state.r2_temporal = None
    st.session_state.overfitting_gap = None
    st.session_state.expanding_results = None
    st.session_state.expanding_mean_r2 = None
    st.session_state.expanding_std_r2 = None
    st.session_state.sliding_results_df = None
    st.session_state.sliding_mean_r2 = None
    st.session_state.sliding_std_r2 = None
    st.session_state.purged_scores = None
    st.session_state.purged_mean_r2 = None
    st.session_state.purged_std_r2 = None
    st.session_state.leakage_inflation = None
    st.session_state.eval_df = None
    st.session_state.regime_perf = None
    st.session_state.learning_curve_diagnosis = None
    st.session_state.final_report_metrics = {}

# Sidebar for navigation
st.sidebar.title("Model Validation Workflow")
page = st.sidebar.selectbox(
    "Choose a validation step:",
    [
        "Introduction",
        "1. Random vs Temporal Split",
        "2. Expanding Window CV",
        "3. Sliding Window CV",
        "4. Purged & Embargo CV",
        "5. Market Regime Analysis",
        "6. Learning Curve",
        "7. Final Report"
    ]
)

# --- Page Rendering Logic ---

if page == "Introduction":
    st.title("Model Validation and Performance Degradation Analysis for Financial Models")
    st.subheader("Introduction: The Portfolio Manager's Dilemma")
    st.markdown(f"**Persona:** Sarah Chen, a seasoned CFA Charterholder and Portfolio Manager at \"Alpha Capital Investments.\"")
    st.markdown(f"**Organization:** Alpha Capital Investments, an institutional asset manager with a strong focus on quantitative strategies and rigorous risk management.")
    st.markdown(f"Sarah is responsible for a multi-billion dollar equity portfolio. Her team frequently develops and deploys quantitative models for investment decisions, such as equity selection and portfolio optimization. Recently, a new equity factor model, developed by a junior quant, showed impressive $R^2$ values during in-sample testing. However, Sarah, with her years of experience and deep understanding of financial markets, knows that in-sample performance can be a deceptive illusion. She's concerned about the \"overfitting gap\"—the difference between apparent and true model performance—which can lead to models that look great on paper but fail catastrophically during real-world market stress or regime changes.")
    st.markdown(f"Her primary objective is to thoroughly validate this new model to ensure it will perform reliably in forward-looking scenarios and to identify potential \"ticking time bombs\" that could erode capital. This rigorous validation is not just good practice; it's a professional and ethical obligation under CFA Standard V(A) (Diligence and Reasonable Basis) and aligns with the firm's model risk management (SR 11-7) protocols. This application walks through the step-by-step validation process Sarah employs.")
    st.markdown(f"---")
    st.markdown(f"### Setup: Data Generation")
    st.markdown(f"Before we begin our validation journey, we'll generate a simulated financial time-series dataset to work with.")

    if st.button("Generate Simulated Financial Data"):
        X, y, dates_aligned, model_config = generate_synthetic_financial_data()
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.dates_aligned = dates_aligned
        st.session_state.model_config = model_config
        st.session_state.data_generated = True
        st.success("Simulated data generated and stored in session state!")
        st.write(f"Dataset shape: X={X.shape}, y={y.shape}")
        st.write(f"Data period: {dates_aligned.min().strftime('%Y-%m-%d')} to {dates_aligned.max().strftime('%Y-%m-%d')}")
    elif st.session_state.data_generated:
        st.info("Simulated data already generated.")
        st.write(f"Dataset shape: X={st.session_state.X.shape}, y={st.session_state.y.shape}")
        st.write(f"Data period: {st.session_state.dates_aligned.min().strftime('%Y-%m-%d')} to {st.session_state.dates_aligned.max().strftime('%Y-%m-%d')}")

elif page == "1. Random vs Temporal Split":
    st.title("1. The Peril of Random Splits: Unveiling the Overfitting Gap")
    st.markdown(f"### Story + Context + Real-World Relevance")
    st.markdown(f"Sarah has received the preliminary report for the new equity factor model. The junior quant proudly highlighted an $R^2$ of $0.12$ on a randomly split test set. However, Sarah immediately raises an eyebrow. Financial time-series data is fundamentally different from typical i.i.d. (independent and identically distributed) data, a core assumption often violated in standard machine learning. Stock returns, for instance, exhibit temporal dependence; today's market conditions often influence tomorrow's. A random train/test split can inadvertently place future data points into the training set and past data points into the test set (or vice-versa, creating overlapping information windows), severely inflating performance metrics due to **information leakage** and **look-ahead bias**. This leads to an **overfitting gap** where apparent performance far exceeds true out-of-sample capability.")
    st.markdown(f"Sarah knows that for financial data, preserving the temporal order is paramount. She decides to demonstrate this critical pitfall by comparing the model's performance using a random split versus a strict temporal split.")

    st.markdown(r"The standard $R^2$ (coefficient of determination) is calculated as:")
    st.markdown(r"$$ R^2 = 1 - \frac{{\sum_{{i=1}}^{{n}} (y_i - \hat{{y}}_i)^2}}{{\sum_{{i=1}}^{{n}} (y_i - \bar{{y}})^2}} $$")
    st.markdown(r"where $y_i$ is the actual value, $\hat{{y}}_i$ is the predicted value, and $\bar{{y}}$ is the mean of the actual values.")

    st.markdown(f"For financial time series, using random splits often leads to an artificially high $R^2$ value because:")
    st.markdown(f"1.  **Temporal Dependence:** Future information \"leaks\" into the training set, especially if the target variable involves future returns (e.g., 1-month forward returns).")
    st.markdown(f"2.  **Non-stationarity:** Market regimes change, and models trained on specific periods might not generalize to others. Random splits mix these regimes.")
    st.markdown(f"By comparing $R^2$ from random vs. temporal splits, Sarah quantifies the \"overfitting gap,\" a critical first step to understanding the model's true predictive power. A significant gap indicates severe overfitting.")

    if st.session_state.data_generated:
        if st.button("Run Random vs Temporal Split Analysis"):
            r2_random, r2_temporal, overfitting_gap = evaluate_split_strategy(
                st.session_state.X, st.session_state.y, st.session_state.dates_aligned, st.session_state.model_config
            )
            st.session_state.r2_random = r2_random
            st.session_state.r2_temporal = r2_temporal
            st.session_state.overfitting_gap = overfitting_gap
            st.subheader("--- Model Performance Comparison (Random vs. Temporal Split) ---")
            st.write(f"R2 with RANDOM split: {r2_random:.4f}")
            st.write(f"R2 with TEMPORAL split: {r2_temporal:.4f}")
            st.write(f"Overfitting Gap (Random R2 - Temporal R2): {overfitting_gap:.4f}")
            if overfitting_gap > 0.05:
                st.warning("Warning: Significant overfitting gap detected! Random splitting artificially inflated performance.")
            st.success("Analysis complete.")
        elif st.session_state.r2_random is not None:
            st.subheader("--- Previous Run: Model Performance Comparison (Random vs. Temporal Split) ---")
            st.write(f"R2 with RANDOM split: {st.session_state.r2_random:.4f}")
            st.write(f"R2 with TEMPORAL split: {st.session_state.r2_temporal:.4f}")
            st.write(f"Overfitting Gap (Random R2 - Temporal R2): {st.session_state.overfitting_gap:.4f}")
            if st.session_state.overfitting_gap > 0.05:
                st.warning("Warning: Significant overfitting gap detected! Random splitting artificially inflated performance.")

    else:
        st.warning("Please generate simulated financial data on the Introduction page first.")

    st.markdown(f"### Explanation of Execution")
    # Addressed SyntaxWarning by using raw string for markdown containing LaTeX math and escaping '%' for regular text if needed.
    st.markdown(r"The output clearly shows a discrepancy between the $R^2$ achieved with a random split and that with a temporal split. The `Overfitting Gap` quantifies this difference. For Sarah, this numerical gap is a visceral demonstration of the danger of relying on standard, non-time-series aware validation. A model might appear to explain $12\%$ of the variance using a random split, but perhaps only $2\%$ when validated correctly by respecting temporal order. This insight confirms her initial suspicions and emphasizes the need for more sophisticated time-series validation techniques. It’s the first step in identifying if a model is a \"ticking time bomb.\"")


elif page == "2. Expanding Window CV":
    st.title("2. Simulating Real-World Deployment with Expanding Window Cross-Validation")
    st.markdown(f"### Story + Context + Real-World Relevance")
    st.markdown(f"Sarah now wants to simulate how the model would perform in a real-time deployment scenario. In practice, models are often retrained periodically (e.g., monthly or quarterly) on all available historical data up to that point. This approach, known as **Expanding Window Walk-Forward Cross-Validation**, mirrors this process. For each fold, the model trains on an ever-growing dataset and then evaluates its performance on the subsequent, unseen period. This provides a more realistic assessment of out-of-sample performance over a sequence of future periods.")

    st.markdown(r"The **Expanding Window** formulation for fold $k$ is given by:")
    st.markdown(r"$$ D^{{(k)}}_{{\text{{train}}}} = \{{(x_t, y_t) : t = 1, \dots, T_k}}\} $$")
    st.markdown(r"$$ D^{{(k)}}_{{\text{{test}}}} = \{{(x_t, y_t) : t = T_k + 1, \dots, T_k + h}}\} $$")
    st.markdown(r"where $T_k = T_{{\min}} + (k-1)h$, $T_{{\min}}$ is the minimum initial training size, and $h$ is the test window length.")
    st.markdown(r"The aggregate performance estimate over $K$ folds is:")
    st.markdown(r"$$ P = \frac{{1}}{{K}} \sum_{{k=1}}^{{K}} L(f^{{(k)}}, D^{{(k)}}_{{\text{{test}}}}) $$")
    st.markdown(r"where $f^{{(k)}}$ is the model trained on $D^{{(k)}}_{{\text{{train}}}}$ and $L$ is the performance metric (e.g., $R^2$, AUC).")

    if st.session_state.data_generated:
        n_splits_expanding = st.slider("Number of splits for Expanding Window CV", 3, 10, 5)
        if st.button("Run Expanding Window CV"):
            results_df, mean_r2, std_r2 = expanding_window_cv(
                st.session_state.X, st.session_state.y, st.session_state.dates_aligned,
                st.session_state.model_config, n_splits=n_splits_expanding
            )
            st.session_state.expanding_results = results_df
            st.session_state.expanding_mean_r2 = mean_r2
            st.session_state.expanding_std_r2 = std_r2

            st.subheader(f"--- Expanding Window Walk-Forward Cross-Validation (n_splits={n_splits_expanding}) ---")
            st.dataframe(results_df)
            st.write(f"Mean CV R2 (Expanding Window): {mean_r2:.4f} +/- {std_r2:.4f}")
            st.success("Expanding Window CV complete.")
        elif st.session_state.expanding_results is not None:
            st.subheader(f"--- Previous Run: Expanding Window Walk-Forward Cross-Validation ---")
            st.dataframe(st.session_state.expanding_results)
            st.write(f"Mean CV R2 (Expanding Window): {st.session_state.expanding_mean_r2:.4f} +/- {st.session_state.expanding_std_r2:.4f}")
    else:
        st.warning("Please generate simulated financial data on the Introduction page first.")

    st.markdown(f"### Explanation of Execution")
    st.markdown(f"The table above details the performance of the model across different expanding window folds. Each fold's $R^2$ gives Sarah a snapshot of the model's predictive power in a specific future period, trained on all data available up to that point. The `Mean CV R2` and its standard deviation (`Std CV R2`) provide an overall picture of the model's average out-of-sample performance and its variability. A low, stable $R^2$ across folds is generally preferred over a high, volatile one. This gives Sarah a much more realistic baseline performance compared to the single, potentially inflated random split $R^2$. She notes down the `Mean CV R2` as a crucial metric for the model's true effectiveness.")


elif page == "3. Sliding Window CV":
    st.title("3. Adapting to Market Dynamics with Sliding Window Cross-Validation")
    st.markdown(f"### Story + Context + Real-World Relevance")
    st.markdown(f"While expanding windows are good for processes assumed to be relatively stable, Sarah knows that financial markets are often non-stationary. Older data might become irrelevant or even detrimental if the underlying data-generating process has changed (e.g., after a major market crisis or technological shift). In such cases, a **Sliding Window Walk-Forward Cross-Validation** approach is more appropriate. Here, the training set size remains fixed, effectively \"sliding\" along the time series, only using the most recent `W` periods for training. This allows the model to adapt more quickly to recent market dynamics but comes at the cost of discarding potentially useful older data and reducing training sample size.")

    st.markdown(r"The **Sliding Window** formulation for fold $k$ replaces the growing training set with a fixed-size window:")
    st.markdown(r"$$ D^{{(k)}}_{{\text{{train}}}} = \{{(x_t, y_t) : t = T_k - W + 1, \dots, T_k}}\} $$")
    st.markdown(r"where $W$ is the fixed window width. This discards old data, adapting to non-stationarity but losing sample size. Sarah needs to visualize the out-of-sample $R^2$ over time to understand model stability.")

    if st.session_state.data_generated:
        train_window_months = st.slider("Training window size (months)", 12, 120, 60)
        test_window_months = st.slider("Test window size (months)", 1, 24, 12)
        if st.button("Run Sliding Window CV"):
            sliding_df, mean_r2_sliding, std_r2_sliding = sliding_window_cv(
                st.session_state.X, st.session_state.y, st.session_state.dates_aligned,
                st.session_state.model_config, train_window_size_months=train_window_months,
                test_window_size_months=test_window_months
            )
            st.session_state.sliding_results_df = sliding_df
            st.session_state.sliding_mean_r2 = mean_r2_sliding
            st.session_state.sliding_std_r2 = std_r2_sliding

            st.subheader(f"--- Sliding Window Walk-Forward Cross-Validation ---")
            st.dataframe(sliding_df)
            
            # Generate plot manually to ensure it displays
            fig_sliding_cv = plt.figure(figsize=(14, 7))
            plt.plot(sliding_df['Test_Period_Start'], sliding_df['R2'], 'b-o', markersize=4, label='Out-of-Sample R2')
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Zero R2 Baseline')
            plt.xlabel('Test Period Start')
            plt.ylabel('Out-of-Sample R2')
            plt.title('Sliding-Window Performance Over Time')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig_sliding_cv)
            
            st.write(f"Mean CV R2 (Sliding Window): {mean_r2_sliding:.4f} +/- {std_r2_sliding:.4f}")
            st.success("Sliding Window CV complete.")
        elif st.session_state.sliding_results_df is not None:
            st.subheader(f"--- Previous Run: Sliding Window Walk-Forward Cross-Validation ---")
            st.dataframe(st.session_state.sliding_results_df)
            
            fig_sliding_cv_rerun = plt.figure(figsize=(14, 7))
            plt.plot(st.session_state.sliding_results_df['Test_Period_Start'], st.session_state.sliding_results_df['R2'], 'b-o', markersize=4, label='Out-of-Sample R2')
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Zero R2 Baseline')
            plt.xlabel('Test Period Start')
            plt.ylabel('Out-of-Sample R2')
            plt.title('Sliding-Window Performance Over Time')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig_sliding_cv_rerun)
            
            st.write(f"Mean CV R2 (Sliding Window): {st.session_state.sliding_mean_r2:.4f} +/- {st.session_state.sliding_std_r2:.4f}")
    else:
        st.warning("Please generate simulated financial data on the Introduction page first.")

    st.markdown(f"### Explanation of Execution")
    st.markdown(f"The plot illustrates how the model's $R^2$ performance varies over time when using a fixed-size training window. Sarah can visually inspect periods where the model performs better or worse. This helps her assess the model's stability and adaptiveness to changing market conditions. If the $R^2$ drops significantly during known periods of market stress, it's a red flag. Compared to the expanding window, the sliding window provides insights into the model's ability to remain relevant in a non-stationary environment. The choice between expanding and sliding windows is a financial judgment: expanding windows assume more stationarity and leverage more data, while sliding windows prioritize recency and adapt to non-stationarity.")


elif page == "4. Purged & Embargo CV":
    st.title("4. Mitigating Information Leakage with Purged and Embargo Cross-Validation")
    st.markdown(f"### Story + Context + Real-World Relevance")
    st.markdown(f"Sarah, as a diligent CFA, understands that financial targets often involve overlapping forward returns. For example, if the target $y_t$ is the return from time $t$ to $t+12$ months, then $y_t$ and $y_{{t+1}}$ share $11$ months of return data. Standard `TimeSeriesSplit` does not account for this, leading to significant **information leakage** when samples in the training set overlap with samples in the test set. This type of leakage can dramatically inflate performance metrics, even in walk-forward validation.")
    st.markdown(f"To combat this, Sarah employs **Purged and Embargo Cross-Validation** (as popularized by Lopez de Prado).")
    st.markdown(f"")
    st.markdown(r"*   **Purge:** Remove training samples that overlap with the test sample's target observation period. If $y_t$ is a $T_p$-period forward return, then any training observation $(x_s, y_s)$ where $s$ is within $T_p$ periods of the test set's start should be removed from the training set.")
    st.markdown(r"$$ D^{{\text{{purged}}}}_{{\text{{train}}}} = \{{(x_t, y_t) \in D_{{\text{{train}}}} : t \leq T_k - T_p}}\} $$")
    st.markdown(r"*   **Embargo:** Introduce a \"buffer\" period immediately following the training set and before the test set. This ensures that information from the test set does not inadvertently \"leak back\" into future training sets through serial correlation or other indirect means.")
    st.markdown(r"$$ D^{{\text{{embargo}}}}_{{\text{{test}}}} = \{{(x_t, y_t) \in D_{{\text{{test}}}} : t > T_k + T_e + 1}}\} $$")
    st.markdown(r"Typical choices are $T_p = \text{{label horizon}}$ (e.g., 12 months for 12-month forward returns) and $T_e = 1-3 \text{{ months}}$ as an additional buffer. Together, these ensure zero information leakage. Sarah wants to quantify the `Leakage Inflation` by comparing standard walk-forward CV $R^2$ to purged and embargoed CV $R^2$.")

    if st.session_state.data_generated and st.session_state.expanding_results is not None:
        n_splits_purged = st.slider("Number of splits for Purged & Embargo CV", 3, 10, 5)
        purge_window_months = st.slider("Purge window (months)", 0, 24, 12)
        embargo_window_months = st.slider("Embargo window (months)", 0, 6, 1)

        if st.button("Run Purged & Embargo CV"):
            # We need the R2 scores from the standard expanding window for leakage inflation calculation
            standard_expanding_r2_series = st.session_state.expanding_results['R2']
            purged_scores, purged_mean_r2, purged_std_r2, leakage_inflation = purged_embargo_walk_forward_cv(
                st.session_state.X, st.session_state.y, st.session_state.dates_aligned,
                st.session_state.model_config, n_splits=n_splits_purged,
                purge_window_months=purge_window_months, embargo_window_months=embargo_window_months,
                expanding_results_r2=standard_expanding_r2_series
            )
            st.session_state.purged_scores = purged_scores
            st.session_state.purged_mean_r2 = purged_mean_r2
            st.session_state.purged_std_r2 = purged_std_r2
            st.session_state.leakage_inflation = leakage_inflation

            st.subheader(f"--- Purged & Embargo Walk-Forward CV (Purge: {purge_window_months}M, Embargo: {embargo_window_months}M) ---")
            st.write(f"Purged+Embargo R2 scores per fold: {np.round(purged_scores, 4)}")
            st.write(f"Mean CV R2 (Purged+Embargo): {purged_mean_r2:.4f} +/- {purged_std_r2:.4f}")
            st.subheader("--- Leakage Analysis ---")
            st.write(f"Mean CV R2 (Standard Expanding Window): {st.session_state.expanding_mean_r2:.4f}")
            st.write(f"Mean CV R2 (Purged+Embargo): {purged_mean_r2:.4f}")
            st.write(f"Leakage Inflation (Standard R2 - Purged R2): {leakage_inflation:.4f}")
            if leakage_inflation > 0.02:
                st.warning("Warning: Significant leakage inflation detected. The model's apparent performance was likely illusory due to data leakage.")
            elif leakage_inflation > 0:
                st.info("Note: Some leakage inflation detected, indicating potential information leakage.")
            else:
                st.success("Good: Minimal or no leakage inflation detected.")
            st.success("Purged & Embargo CV complete.")
        elif st.session_state.purged_mean_r2 is not None:
            st.subheader(f"--- Previous Run: Purged & Embargo Walk-Forward CV ---")
            st.write(f"Purged+Embargo R2 scores per fold: {np.round(st.session_state.purged_scores, 4)}")
            st.write(f"Mean CV R2 (Purged+Embargo): {st.session_state.purged_mean_r2:.4f} +/- {st.session_state.purged_std_r2:.4f}")
            st.subheader("--- Leakage Analysis ---")
            st.write(f"Mean CV R2 (Standard Expanding Window): {st.session_state.expanding_mean_r2:.4f}")
            st.write(f"Mean CV R2 (Purged+Embargo): {st.session_state.purged_mean_r2:.4f}")
            st.write(f"Leakage Inflation (Standard R2 - Purged R2): {st.session_state.leakage_inflation:.4f}")
            if st.session_state.leakage_inflation > 0.02:
                st.warning("Warning: Significant leakage inflation detected. The model's apparent performance was likely illusory due to data leakage.")
            elif st.session_state.leakage_inflation > 0:
                st.info("Note: Some leakage inflation detected, indicating potential information leakage.")
            else:
                st.success("Good: Minimal or no leakage inflation detected.")
    else:
        st.warning("Please generate simulated financial data and run Expanding Window CV first.")


elif page == "5. Market Regime Analysis":
    st.title("5. Stress-Testing Across Market Regimes: Identifying \"Ticking Time Bombs\"")
    st.markdown(f"### Story + Context + Real-World Relevance")
    st.markdown(f"A model might perform well in benign bull markets but collapse during bear markets or periods of high volatility. Sarah needs to understand the model's robustness across different **market regimes** to identify these \"ticking time bombs.\" Relying on average performance metrics (like mean $R^2$) can mask critical weaknesses during periods when accurate predictions are most crucial. For instance, a credit model's performance in a recession is far more important than its performance during an expansion.")
    st.markdown(f"Sarah decides to classify historical market regimes based on common financial indicators: S&P 500 returns for \"bull\" vs. \"bear\" and VIX levels for \"high volatility\" vs. \"low volatility.\" She will then re-evaluate the model's performance (specifically, the distribution of residuals) within each regime. A model that works in bull markets but fails in bear markets, or one whose errors explode during high volatility, is unacceptable for Alpha Capital.")
    st.markdown(f"The regimes are defined as:")
    st.markdown(f"*   **Bull:** 12-month rolling average of S&P 500 returns is positive.")
    st.markdown(f"*   **Bear:** 12-month rolling average of S&P 500 returns is negative.")
    st.markdown(f"*   **High Volatility:** VIX index is above its median.")
    st.markdown(f"*   **Low Volatility:** VIX index is below its median.")
    st.markdown(f"These are then combined into four primary regimes: Bull-LowVol, Bull-HighVol, Bear-LowVol, and Bear-HighVol.")

    if st.session_state.data_generated:
        if st.button("Run Market Regime Analysis"):
            eval_df, regime_perf, fig1, fig2 = calculate_regime_performance( # Fixed: Renamed function call
                st.session_state.X, st.session_state.y, st.session_state.dates_aligned, st.session_state.model_config
            )
            st.session_state.eval_df = eval_df
            st.session_state.regime_perf = regime_perf

            st.subheader("--- Market Regimes Distribution ---")
            st.write(eval_df['regime'].value_counts())
            st.subheader("--- Evaluation DataFrame Head with Regimes ---")
            st.dataframe(eval_df.head())
            st.subheader("--- Regime-Conditional Performance Metrics (Full Data) ---")
            st.dataframe(regime_perf)
            st.subheader("--- Regime-Conditional Performance Visualizations ---")
            st.pyplot(fig1) 
            st.pyplot(fig2) 
            st.success("Market Regime Analysis complete.")
        elif st.session_state.regime_perf is not None:
            st.subheader("--- Previous Run: Market Regimes Distribution ---")
            st.write(st.session_state.eval_df['regime'].value_counts())
            st.subheader("--- Previous Run: Evaluation DataFrame Head with Regimes ---")
            st.dataframe(st.session_state.eval_df.head())
            st.subheader("--- Previous Run: Regime-Conditional Performance Metrics (Full Data) ---")
            st.dataframe(st.session_state.regime_perf)
            st.subheader("--- Previous Run: Regime-Conditional Performance Visualizations ---")
            # Re-generate plots if needed for display on revisiting the page
            _, _, fig1_re, fig2_re = calculate_regime_performance( # Fixed: Renamed function call
                st.session_state.X, st.session_state.y, st.session_state.dates_aligned, st.session_state.model_config,
                return_figures_only=True, existing_eval_df=st.session_state.eval_df
            )
            st.pyplot(fig1_re)
            st.pyplot(fig2_re)
    else:
        st.warning("Please generate simulated financial data on the Introduction page first.")

    st.markdown(f"### Explanation of Execution")
    st.markdown(f"The `Regime-Conditional Performance Metrics` table and the box plots of residuals are extremely insightful for Sarah. She can now clearly see how the model's $R^2$ changes across different market conditions and observe the distribution of prediction errors. For instance, if the $R^2$ is high in 'Bull-LowVol' but near zero or negative in 'Bear-HighVol', it signals a critical weakness. A wider box plot of residuals (indicating larger errors) or a median shifted significantly from zero in 'Bear-HighVol' would be a major warning. This analysis directly informs Sarah if the model is robust enough for all market environments or if it's a fair-weather performer, helping her prevent capital loss during periods of market stress.")


elif page == "6. Learning Curve":
    st.title("6. Diagnosing Bias-Variance with Learning Curves")
    st.markdown(f"### Story + Context + Real-World Relevance")
    st.markdown(f"Sarah understands that a model's performance is affected by its complexity and the amount of training data. To diagnose if the factor model is suffering from **high bias (underfitting)** or **high variance (overfitting)**, and whether acquiring more data would improve its performance, she uses **Learning Curves**. A learning curve plots the training and cross-validation scores as a function of the training set size.")
    st.markdown(f"The expected behaviors are:")
    st.markdown(f"*   **High bias (underfitting):** Both training and validation scores are low and converge at a low value. More data does not help; the model is too simple to capture the underlying patterns.")
    st.markdown(f"*   **High variance (overfitting):** Training score is high, but validation score is low, with a significant gap between them. As training data increases, the gap narrows, and the validation score eventually rises. More data helps.")
    st.markdown(f"*   **Optimal complexity:** Training score is moderately high, validation score is close to it, and the gap is small and stable.")
    st.markdown(f"For financial models, high variance is a common pathology (too many features relative to samples, or model is too flexible), while high bias can occur with overly restrictive models (e.g., linear models on non-linear data). Sarah wants to understand where this model stands to inform potential next steps: feature engineering, model complexity adjustment, or data acquisition.")

    if st.session_state.data_generated:
        if st.button("Run Learning Curve Analysis"):
            tscv_lc = TimeSeriesSplit(n_splits=5)
            learning_curve_diagnosis, fig_lc = plot_learning_curve(
                st.session_state.X, st.session_state.y, st.session_state.model_config,
                tscv_lc, scoring_metric='r2'
            )
            st.session_state.learning_curve_diagnosis = learning_curve_diagnosis

            st.subheader("--- Learning Curve: Bias-Variance Diagnosis ---")
            st.pyplot(fig_lc) 
            st.write(f"Final Training Score: {learning_curve_diagnosis['final_train_score']:.4f}")
            st.write(f"Final Cross-validation Score: {learning_curve_diagnosis['final_test_score']:.4f}")
            st.write(f"Gap at Max Training Data: {learning_curve_diagnosis['gap_at_max_data']:.4f}")
            st.write(f"Recommendation: {learning_curve_diagnosis['recommendation']}")
            st.success("Learning Curve Analysis complete.")
        elif st.session_state.learning_curve_diagnosis is not None:
            st.subheader("--- Previous Run: Learning Curve: Bias-Variance Diagnosis ---")
            tscv_lc_re = TimeSeriesSplit(n_splits=5)
            _, fig_lc_re = plot_learning_curve(
                st.session_state.X, st.session_state.y, st.session_state.model_config,
                tscv_lc_re, scoring_metric='r2', return_figure_only=True
            )
            st.pyplot(fig_lc_re)
            diagnosis = st.session_state.learning_curve_diagnosis
            st.write(f"Final Training Score: {diagnosis['final_train_score']:.4f}")
            st.write(f"Final Cross-validation Score: {diagnosis['final_test_score']:.4f}")
            st.write(f"Gap at Max Training Data: {diagnosis['gap_at_max_data']:.4f}")
            st.write(f"Recommendation: {diagnosis['recommendation']}")
    else:
        st.warning("Please generate simulated financial data on the Introduction page first.")

    st.markdown(f"### Explanation of Execution")
    st.markdown(f"The learning curve provides a visual and quantitative diagnosis of the model's bias-variance tradeoff. Sarah can see if the training and cross-validation scores converge and at what level. If the two curves are far apart, it suggests high variance (overfitting), implying the model is too complex for the given data or needs more data. If both curves are low and flat, it indicates high bias (underfitting), meaning the model is too simple. This helps Sarah make informed decisions: if it's high variance, she might consider regularization or increasing the data size; if it's high bias, she might explore more complex features or model architectures. This is a crucial step in understanding the model's fundamental limitations and potential for improvement.")


elif page == "7. Final Report":
    st.title("7. Quantifying Model Stability and Overfitting: The Validation Report")
    st.markdown(f"### Story + Context + Real-World Relevance")
    st.markdown(f"After running multiple rigorous validation tests, Sarah now needs to synthesize all findings into a concise, actionable \"Model Validation Report.\" This report is the final deliverable for Alpha Capital's Investment Committee and the Model Risk Management (MRM) team. It must clearly communicate the model's strengths, weaknesses, and a go/no-go recommendation based on quantified metrics of stability and overfitting.Key metrics Sarah needs to present include:")
    st.markdown(f"1.  **Overfitting Ratio (OFR):** Measures the relative degradation from in-sample to out-of-sample performance.")
    st.markdown(r"$$\text{{OFR}} = \frac{{P_{{\text{{in-sample}}}} - P_{{\text{{out-of-sample}}}}}}{{P_{{\text{{in-sample}}}}}} $$")
    st.markdown(r"where $P_{{\text{{in-sample}}}}$ is the $R^2$ from a random split (representing potentially inflated in-sample performance) and $P_{{\text{{out-of-sample}}}} $ is the $R^2$ from a robust temporal split (e.g., the mean of purged CV or expanding window CV).")
    st.markdown(f"*   OFR $\approx 0$: No overfitting.")
    st.markdown(f"*   OFR $> 0.5$: Significant overfitting, model is essentially memorizing.")
    st.markdown(f"2.  **Performance Stability Ratio (PSR):** Measures the consistency of out-of-sample performance across different folds/regimes.")
    st.markdown(r"$$\text{{PSR}} = \frac{{\min_{{k}} P^{{(k)}}_{{\text{{test}}}}}}{{\max_{{k}} P^{{(k)}}_{{\text{{test}}}}}} $$")
    st.markdown(r"where $P^{{(k)}}_{{\text{{test}}}}$ is the test performance (e.g., $R^2$) for fold $k$.")
    st.markdown(f"*   PSR $\approx 1$: Performance is stable.")
    st.markdown(f"*   PSR $< 0$: Model performs well in some periods but has negative performance in others (e.g., negative $R^2$)—a red flag.")
    st.markdown(f"Sarah will gather all calculated metrics, regime performance, and the learning curve diagnosis into a structured report template, ready for presentation.")

    if st.session_state.data_generated and st.session_state.r2_random is not None and st.session_state.purged_mean_r2 is not None and st.session_state.regime_perf is not None and st.session_state.learning_curve_diagnosis is not None:
        if st.button("Generate Model Validation Report"):
            report_metrics = {
                'r2_random': st.session_state.r2_random,
                'r2_temporal': st.session_state.r2_temporal,
                'overfitting_gap': st.session_state.overfitting_gap,
                'expanding_mean_r2': st.session_state.expanding_mean_r2,
                'expanding_std_r2': st.session_state.expanding_std_r2,
                'sliding_mean_r2': st.session_state.sliding_mean_r2,
                'sliding_std_r2': st.session_state.sliding_std_r2,
                'purged_scores': st.session_state.purged_scores,
                'purged_mean_r2': st.session_state.purged_mean_r2,
                'purged_std_r2': st.session_state.purged_std_r2,
                'leakage_inflation': st.session_state.leakage_inflation,
                'regime_perf': st.session_state.regime_perf,
                'learning_curve_diagnosis': st.session_state.learning_curve_diagnosis,
            }
            st.session_state.final_report_metrics = report_metrics 

            final_report_string = generate_model_validation_report(**report_metrics)
            st.text(final_report_string)

            st.subheader("Final Go/No-Go Recommendation")
            recommendation = st.radio(
                "Based on the report, what is your recommendation?",
                ["Approved for production", "Requires further development", "Rejected"]
            )
            st.info(f"You have recommended: **{recommendation}**")
            st.text_area("Provide rationale for your recommendation:", height=150)

            st.success("Model Validation Report generated.")
        elif st.session_state.final_report_metrics:
            st.subheader("--- Previous Run: Model Validation Report ---")
            final_report_string = generate_model_validation_report(**st.session_state.final_report_metrics)
            st.text(final_report_string)
            st.subheader("Final Go/No-Go Recommendation")
            st.info("Recommendation captured from previous interaction.")

    else:
        st.warning("Please complete all previous validation steps to generate the full report.")

    st.markdown(f"### Explanation of Execution")
    st.markdown(f"This comprehensive `Model Validation Report` provides Sarah with a holistic view of the model's reliability. Each metric (Overfitting Ratio, Performance Stability Ratio, Leakage Inflation, Regime Spread) is explicitly calculated and interpreted. This allows her to quantify the risks associated with the model's deployment. The report serves as the basis for a data-driven investment decision, fulfilling her ethical obligations as a CFA Charterholder and aligning with Alpha Capital's rigorous risk management framework. For instance, if the Overfitting Ratio is high and the Performance Stability Ratio is low, coupled with poor performance in 'Bear-HighVol' regimes, Sarah will likely recommend against deploying the model without significant further development or rejection. This is how quantitative validation directly translates into prudent capital allocation and risk management in a financial institution.")

