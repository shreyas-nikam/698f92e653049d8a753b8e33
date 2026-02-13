
from streamlit.testing.v1 import AppTest
import pytest
import pandas as pd
import numpy as np
import datetime

# Note: For these tests to run successfully, a 'source.py' file containing
# the functions imported by 'app.py' (e.g., generate_simulated_financial_data,
# evaluate_split_strategy, etc.) must be present in the same directory or accessible
# via Python path. These functions are assumed to be implemented and functional.

def test_introduction_page_initial_state():
    """
    Tests the initial state of the Introduction page, ensuring data has not yet been generated.
    """
    at = AppTest.from_file("app.py").run()
    # Verify initial session state: data_generated should be False
    assert not at.session_state["data_generated"]
    # Verify the "Generate Simulated Financial Data" button is present
    assert "Generate Simulated Financial Data" in at.button[0].label
    # Verify no info or success messages are displayed initially
    assert len(at.info) == 0
    assert len(at.success) == 0

def test_introduction_page_data_generation():
    """
    Tests the data generation functionality on the Introduction page.
    """
    at = AppTest.from_file("app.py").run()
    at.button[0].click().run() # Click "Generate Simulated Financial Data"

    # Verify data is generated and stored in session state
    assert at.session_state["data_generated"]
    assert at.session_state["X"] is not None
    assert at.session_state["y"] is not None
    assert at.session_state["dates_aligned"] is not None
    assert at.session_state["model_config"] is not None

    # Verify the success message and data details are displayed
    assert at.success[0].value == "Simulated data generated and stored in session state!"
    assert "Dataset shape: X=" in at.write[0].value
    assert "Data period:" in at.write[1].value

    # Test re-running the app or navigating back to the Introduction page when data is already generated
    at.run() # Simulate re-running the app
    assert at.info[0].value == "Simulated data already generated."

def test_random_vs_temporal_split_page():
    """
    Tests the "Random vs Temporal Split" page, including running the analysis
    and checking for the overfitting gap warning.
    """
    # First, ensure data is generated to enable the analysis button
    at = AppTest.from_file("app.py").run()
    at.button[0].click().run() # Generate Simulated Financial Data

    # Navigate to the "1. Random vs Temporal Split" page
    at.selectbox[0].set_value("1. Random vs Temporal Split").run()

    # Verify the analysis button is present
    assert at.button[0].label == "Run Random vs Temporal Split Analysis"

    # Click the analysis button
    at.button[0].click().run()

    # Verify results are displayed and stored in session state
    assert at.session_state["r2_random"] is not None
    assert at.session_state["r2_temporal"] is not None
    assert at.session_state["overfitting_gap"] is not None

    assert at.subheader[1].value == "--- Model Performance Comparison (Random vs. Temporal Split) ---"
    assert f"R2 with RANDOM split: {at.session_state['r2_random']:.4f}" in at.write[0].value
    assert f"R2 with TEMPORAL split: {at.session_state['r2_temporal']:.4f}" in at.write[1].value
    assert f"Overfitting Gap (Random R2 - Temporal R2): {at.session_state['overfitting_gap']:.4f}" in at.write[2].value
    assert at.success[0].value == "Analysis complete."

    # Test the warning for significant overfitting if applicable based on simulated data
    if at.session_state['overfitting_gap'] > 0.05:
        assert at.warning[0].value == "Warning: Significant overfitting gap detected! Random splitting artificially inflated performance."
    else:
        assert len(at.warning) == 0 # Ensure no warning if gap is not significant

    # Verify the "Previous Run" state by re-running the app
    at.run()
    assert at.subheader[1].value == "--- Previous Run: Model Performance Comparison (Random vs. Temporal Split) ---"
    assert f"R2 with RANDOM split: {at.session_state['r2_random']:.4f}" in at.write[0].value

def test_expanding_window_cv_page():
    """
    Tests the "Expanding Window CV" page, including slider interaction and analysis execution.
    """
    # First, ensure data is generated
    at = AppTest.from_file("app.py").run()
    at.button[0].click().run() # Generate Simulated Financial Data

    # Navigate to the "2. Expanding Window CV" page
    at.selectbox[0].set_value("2. Expanding Window CV").run()

    # Adjust the number of splits slider
    at.slider[0].set_value(7).run() # n_splits_expanding = 7

    # Click the analysis button
    at.button[0].click().run()

    # Verify results are displayed and stored in session state
    assert at.session_state["expanding_results"] is not None
    assert at.session_state["expanding_mean_r2"] is not None
    assert at.session_state["expanding_std_r2"] is not None

    assert at.subheader[1].value == f"--- Expanding Window Walk-Forward Cross-Validation (n_splits=7) ---"
    assert not at.dataframe[0].empty # Check if the dataframe is displayed and not empty
    assert f"Mean CV R2 (Expanding Window): {at.session_state['expanding_mean_r2']:.4f} +/- {at.session_state['expanding_std_r2']:.4f}" in at.write[0].value
    assert at.success[0].value == "Expanding Window CV complete."

    # Verify the "Previous Run" state
    at.run()
    assert at.subheader[1].value == f"--- Previous Run: Expanding Window Walk-Forward Cross-Validation ---"
    assert not at.dataframe[0].empty

def test_sliding_window_cv_page():
    """
    Tests the "Sliding Window CV" page, including slider interaction, analysis execution, and plot verification.
    """
    # First, ensure data is generated
    at = AppTest.from_file("app.py").run()
    at.button[0].click().run() # Generate Simulated Financial Data

    # Navigate to the "3. Sliding Window CV" page
    at.selectbox[0].set_value("3. Sliding Window CV").run()

    # Adjust sliders for training and test window sizes
    at.slider[0].set_value(24).run() # train_window_months = 24
    at.slider[1].set_value(6).run()  # test_window_months = 6

    # Click the analysis button
    at.button[0].click().run()

    # Verify results are displayed and stored in session state
    assert at.session_state["sliding_results_df"] is not None
    assert at.session_state["sliding_mean_r2"] is not None
    assert at.session_state["sliding_std_r2"] is not None

    assert at.subheader[1].value == f"--- Sliding Window Walk-Forward Cross-Validation ---"
    assert not at.dataframe[0].empty
    assert len(at.pyplot) > 0 # Verify that a plot was generated and displayed
    assert f"Mean CV R2 (Sliding Window): {at.session_state['sliding_mean_r2']:.4f} +/- {at.session_state['sliding_std_r2']:.4f}" in at.write[0].value
    assert at.success[0].value == "Sliding Window CV complete."

    # Verify the "Previous Run" state
    at.run()
    assert at.subheader[1].value == f"--- Previous Run: Sliding Window Walk-Forward Cross-Validation ---"
    assert not at.dataframe[0].empty
    assert len(at.pyplot) > 0 # Verify plot is still present

def test_purged_embargo_cv_page():
    """
    Tests the "Purged & Embargo CV" page, including slider interaction, analysis execution,
    and checking for leakage inflation messages.
    """
    # Ensure data is generated and Expanding Window CV is run first, as it's used for comparison
    at = AppTest.from_file("app.py").run()
    at.button[0].click().run() # Generate Simulated Financial Data
    at.selectbox[0].set_value("2. Expanding Window CV").run()
    at.button[0].click().run() # Run Expanding Window CV

    # Navigate to the "4. Purged & Embargo CV" page
    at.selectbox[0].set_value("4. Purged & Embargo CV").run()

    # Adjust sliders for splits, purge, and embargo windows
    at.slider[0].set_value(7).run()  # n_splits_purged = 7
    at.slider[1].set_value(6).run()  # purge_window_months = 6
    at.slider[2].set_value(2).run()  # embargo_window_months = 2

    # Click the analysis button
    at.button[0].click().run()

    # Verify results are displayed and stored in session state
    assert at.session_state["purged_scores"] is not None
    assert at.session_state["purged_mean_r2"] is not None
    assert at.session_state["purged_std_r2"] is not None
    assert at.session_state["leakage_inflation"] is not None

    assert at.subheader[1].value == f"--- Purged & Embargo Walk-Forward CV (Purge: 6M, Embargo: 2M) ---"
    assert f"Mean CV R2 (Purged+Embargo): {at.session_state['purged_mean_r2']:.4f} +/- {at.session_state['purged_std_r2']:.4f}" in at.write[1].value
    assert at.subheader[2].value == "--- Leakage Analysis ---"
    assert f"Leakage Inflation (Standard R2 - Purged R2): {at.session_state['leakage_inflation']:.4f}" in at.write[4].value
    assert at.success[0].value == "Purged & Embargo CV complete."

    # Test leakage warning/info/success messages based on the calculated leakage inflation
    if at.session_state['leakage_inflation'] > 0.02:
        assert at.warning[0].value == "Warning: Significant leakage inflation detected. The model's apparent performance was likely illusory due to data leakage."
    elif at.session_state['leakage_inflation'] > 0:
        assert at.info[0].value == "Note: Some leakage inflation detected, indicating potential information leakage."
    else:
        assert at.success[1].value == "Good: Minimal or no leakage inflation detected." # Second success element

    # Verify the "Previous Run" state
    at.run()
    assert at.subheader[1].value == f"--- Previous Run: Purged & Embargo Walk-Forward CV ---"
    assert f"Mean CV R2 (Purged+Embargo): {at.session_state['purged_mean_r2']:.4f} +/- {at.session_state['purged_std_r2']:.4f}" in at.write[1].value

def test_market_regime_analysis_page():
    """
    Tests the "Market Regime Analysis" page, verifying the execution of analysis
    and the display of results and plots.
    """
    # Ensure data is generated
    at = AppTest.from_file("app.py").run()
    at.button[0].click().run() # Generate Simulated Financial Data

    # Navigate to the "5. Market Regime Analysis" page
    at.selectbox[0].set_value("5. Market Regime Analysis").run()

    # Click the analysis button
    at.button[0].click().run()

    # Verify results are displayed and stored in session state
    assert at.session_state["eval_df"] is not None
    assert at.session_state["regime_perf"] is not None

    assert at.subheader[1].value == "--- Market Regimes Distribution ---"
    assert at.subheader[2].value == "--- Evaluation DataFrame Head with Regimes ---"
    assert at.subheader[3].value == "--- Regime-Conditional Performance Metrics (Full Data) ---"
    assert at.subheader[4].value == "--- Regime-Conditional Performance Visualizations ---"

    # Check for presence of output elements
    assert not at.write[0].empty # Output of eval_df['regime'].value_counts()
    assert not at.dataframe[0].empty # eval_df.head()
    assert not at.dataframe[1].empty # regime_perf
    assert len(at.pyplot) >= 2 # Verify both plots (fig1, fig2) exist
    assert at.success[0].value == "Market Regime Analysis complete."

    # Verify the "Previous Run" state
    at.run()
    assert at.subheader[1].value == "--- Previous Run: Market Regimes Distribution ---"
    assert len(at.pyplot) >= 2 # Plots should be re-generated for display

def test_learning_curve_page():
    """
    Tests the "Learning Curve" page, verifying analysis execution and diagnosis display.
    """
    # Ensure data is generated
    at = AppTest.from_file("app.py").run()
    at.button[0].click().run() # Generate Simulated Financial Data

    # Navigate to the "6. Learning Curve" page
    at.selectbox[0].set_value("6. Learning Curve").run()

    # Click the analysis button
    at.button[0].click().run()

    # Verify results are displayed and stored in session state
    assert at.session_state["learning_curve_diagnosis"] is not None

    assert at.subheader[1].value == "--- Learning Curve: Bias-Variance Diagnosis ---"
    assert len(at.pyplot) > 0 # Verify the learning curve plot exists
    assert f"Final Training Score: {at.session_state['learning_curve_diagnosis']['final_train_score']:.4f}" in at.write[0].value
    assert f"Final Cross-validation Score: {at.session_state['learning_curve_diagnosis']['final_test_score']:.4f}" in at.write[1].value
    assert f"Gap at Max Training Data: {at.session_state['learning_curve_diagnosis']['gap_at_max_data']:.4f}" in at.write[2].value
    assert f"Recommendation: {at.session_state['learning_curve_diagnosis']['recommendation']}" in at.write[3].value
    assert at.success[0].value == "Learning Curve Analysis complete."

    # Verify the "Previous Run" state
    at.run()
    assert at.subheader[1].value == "--- Previous Run: Learning Curve: Bias-Variance Diagnosis ---"
    assert len(at.pyplot) > 0
    assert f"Recommendation: {at.session_state['learning_curve_diagnosis']['recommendation']}" in at.write[3].value

def test_final_report_page():
    """
    Tests the "Final Report" page, ensuring all preceding steps are completed,
    report generation, and interaction with recommendation components.
    """
    # To enable the final report generation, all previous steps must populate session state.
    # We simulate this by running through each relevant page.
    at = AppTest.from_file("app.py").run()
    at.button[0].click().run() # Generate Simulated Financial Data (Introduction)

    at.selectbox[0].set_value("1. Random vs Temporal Split").run()
    at.button[0].click().run() # Run Random vs Temporal Split Analysis

    at.selectbox[0].set_value("2. Expanding Window CV").run()
    at.slider[0].set_value(5).run() # Set a value for n_splits_expanding to prevent potential default issues
    at.button[0].click().run() # Run Expanding Window CV

    at.selectbox[0].set_value("3. Sliding Window CV").run()
    at.slider[0].set_value(24).run()
    at.slider[1].set_value(6).run()
    at.button[0].click().run() # Run Sliding Window CV

    at.selectbox[0].set_value("4. Purged & Embargo CV").run()
    at.slider[0].set_value(5).run() # Set a value for n_splits_purged
    at.slider[1].set_value(12).run() # Set purge_window_months
    at.slider[2].set_value(1).run() # Set embargo_window_months
    at.button[0].click().run() # Run Purged & Embargo CV

    at.selectbox[0].set_value("5. Market Regime Analysis").run()
    at.button[0].click().run() # Run Market Regime Analysis

    at.selectbox[0].set_value("6. Learning Curve").run()
    at.button[0].click().run() # Run Learning Curve Analysis

    # Now navigate to the "7. Final Report" page
    at.selectbox[0].set_value("7. Final Report").run()

    # Click to generate the report
    at.button[0].click().run()

    # Verify report is generated and session state is updated
    assert at.session_state["final_report_metrics"] is not None
    assert "Model Validation Report" in at.text[0].value # The report string is typically in st.text

    # Interact with the recommendation radio button and text area
    assert at.radio[0].options == ["Approved for production", "Requires further development", "Rejected"]
    at.radio[0].set_value("Requires further development").run()
    assert at.info[0].value == "You have recommended: **Requires further development**"

    at.text_area[0].set_value("Needs more feature engineering and testing in bear markets.").run()
    assert at.text_area[0].value == "Needs more feature engineering and testing in bear markets."

    assert at.success[0].value == "Model Validation Report generated."

    # Verify the "Previous Run" state
    at.run()
    assert at.subheader[1].value == "--- Previous Run: Model Validation Report ---"
    assert "Recommendation captured from previous interaction." in at.info[0].value
