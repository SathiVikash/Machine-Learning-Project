"""
Explainable AI (XAI) Module using SHAP (SHapley Additive exPlanations).
Implements:
1. TreeExplainer for XGBoost / Random Forest
2. Global Feature Importance (Beeswarm, Bar plots)
3. Non-linear Threshold & Interaction Discovery (Dependence plots: NDVI, NDBI, Water buffer)
4. Local Case Explanations (Waterfall plots for specific high-risk wards)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from src.config import EXPOSURE_FEATURES, FIGURES_DIR, OUTPUTS_DIR

def run_shap_analysis(model, df, region_key="trichy_delta", sample_size=1000):
    """
    Computes SHAP values on the model and exports publication-ready explainability plots.
    """
    X = df[EXPOSURE_FEATURES].copy()
    
    # Subsample if dataset is large to maintain fast explainability computation
    if len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=42)
    else:
        X_sample = X.copy()

    # Initialize TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    # 1. Global Mean Absolute SHAP Importance Table
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        "Feature": EXPOSURE_FEATURES,
        "Mean_Abs_SHAP_degC": np.round(mean_abs_shap, 4)
    }).sort_values(by="Mean_Abs_SHAP_degC", ascending=False)
    
    importance_csv = OUTPUTS_DIR / f"{region_key}_shap_feature_importance.csv"
    importance_df.to_csv(importance_csv, index=False)

    # Configure plotting style
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8

    # 2. SHAP Summary Beeswarm Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, max_display=10, show=False)
    plt.title(f"SHAP Feature Impact on Surface Temperature ({region_key})", fontsize=13, pad=12)
    plt.tight_layout()
    beeswarm_path = FIGURES_DIR / f"{region_key}_shap_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 3. SHAP Feature Importance Bar Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.bar(shap_values, max_display=10, show=False)
    plt.title(f"Global Factor Contribution Ranking ({region_key})", fontsize=13, pad=12)
    plt.tight_layout()
    bar_path = FIGURES_DIR / f"{region_key}_shap_bar.png"
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Non-Linear Threshold Dependence Plots
    # A. NDVI Dependence (Green cooling threshold)
    if "NDVI" in EXPOSURE_FEATURES:
        fig, ax = plt.subplots(figsize=(7, 5))
        shap.plots.scatter(shap_values[:, "NDVI"], color=shap_values[:, "NDBI"], show=False)
        plt.title("Non-linear Cooling Effect of Vegetation (NDVI vs. SHAP)", fontsize=12)
        plt.axhline(0, color="gray", linestyle="--", alpha=0.7)
        plt.tight_layout()
        ndvi_dep_path = FIGURES_DIR / f"{region_key}_shap_dependence_ndvi.png"
        plt.savefig(ndvi_dep_path, dpi=300, bbox_inches="tight")
        plt.close()

    # B. NDBI Dependence (Impervious heating threshold)
    if "NDBI" in EXPOSURE_FEATURES:
        fig, ax = plt.subplots(figsize=(7, 5))
        shap.plots.scatter(shap_values[:, "NDBI"], color=shap_values[:, "Building_Density_pct"], show=False)
        plt.title("Thermal Heating Impact of Built-up Density (NDBI vs. SHAP)", fontsize=12)
        plt.axhline(0, color="gray", linestyle="--", alpha=0.7)
        plt.tight_layout()
        ndbi_dep_path = FIGURES_DIR / f"{region_key}_shap_dependence_ndbi.png"
        plt.savefig(ndbi_dep_path, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"[{region_key}] Exported SHAP figures and importance table -> {FIGURES_DIR}")

    return {
        "importance_df": importance_df,
        "explainer": explainer,
        "shap_values": shap_values,
        "X_sample": X_sample
    }

def explain_single_ward_or_cell(explainer, input_row, feature_names=EXPOSURE_FEATURES):
    """
    Produces a local explanation for a specific location / administrative ward.
    """
    if isinstance(input_row, pd.Series):
        X_row = input_row[feature_names].values.reshape(1, -1)
    else:
        X_row = np.array(input_row).reshape(1, -1)
        
    shap_vals = explainer(X_row)
    return shap_vals
