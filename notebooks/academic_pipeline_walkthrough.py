# %% [markdown]
# # Explainable Machine Learning for Urban Heat Vulnerability Assessment
# ## Academic Pipeline & Research Walkthrough
# 
# **Case Studies:**
# 1. **Tiruchirappalli (Cauvery Delta Gateway)** - Delta urban-agricultural fringe
# 2. **Chennai Metropolitan Area** - Dense coastal metropolis
#
# This interactive notebook walks through data exploration, Spatial Block Cross-Validation,
# SHAP explainability, and Administrative Ward Risk Prioritization.

# %% [markdown]
# ### 1. Environment & Imports

# %%
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set paths
BASE_DIR = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path.cwd()
sys.path.insert(0, str(BASE_DIR))

from src.config import REGIONS, EXPOSURE_FEATURES, TARGET_VARIABLE, VULNERABILITY_FEATURES, RISK_COLORS
from src.data_generator import generate_synthetic_region_data
from src.models import train_and_evaluate_models
from src.vulnerability import compute_vulnerability_and_risk
from src.explainability import run_shap_analysis
from src.ward_aggregation import aggregate_to_administrative_wards

print("Environment loaded successfully. Working Directory:", BASE_DIR)

# %% [markdown]
# ### 2. Data Exploration & Feature Harmonization
# We explore the high-resolution grid dataset (250m x 250m) for the Cauvery Delta region.

# %%
region_key = "trichy_delta"
df_trichy, wards_geojson = generate_synthetic_region_data(region_key)

print(f"Generated {len(df_trichy)} grid cells across {df_trichy['ward_id'].nunique()} wards.")
df_trichy.head()

# %% [markdown]
# ### 3. Microclimate & Thermal Distributions
# Let's inspect the distribution of Land Surface Temperature (LST) and vegetation (NDVI).

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

sns.histplot(df_trichy["LST_Celsius"], bins=30, kde=True, color="#e6550d", ax=axes[0])
axes[0].set_title("Land Surface Temperature (°C)")
axes[0].set_xlabel("LST (°C)")

sns.histplot(df_trichy["NDVI"], bins=30, kde=True, color="#31a354", ax=axes[1])
axes[1].set_title("Normalized Difference Vegetation Index (NDVI)")
axes[1].set_xlabel("NDVI")

sns.histplot(df_trichy["NDBI"], bins=30, kde=True, color="#756bb1", ax=axes[2])
axes[2].set_title("Normalized Difference Built-up Index (NDBI)")
axes[2].set_xlabel("NDBI")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4. Spatial Block Cross-Validation & Model Training
# We train Linear Regression (Baseline), Random Forest, and XGBoost using **Spatial Block Cross-Validation** to prevent spatial autocorrelation data leakage.

# %%
model_results = train_and_evaluate_models(df_trichy, region_key)
metrics_table = model_results["metrics_df"]
best_model = model_results["best_model"]
tree_model = model_results["tree_model"]
df_pred = model_results["df_with_predictions"]

print("\n--- Spatial Block Cross-Validation Results ---")
print(metrics_table.to_string(index=False))

# %% [markdown]
# ### 5. PCA Dimensionality Reduction
# Analyzes multicollinearity and feature redundancy across spectral and urban form variables.

# %%
explained_var = model_results["pca_variance"]
plt.figure(figsize=(7, 4))
plt.bar(range(1, len(explained_var) + 1), explained_var * 100, color="#2b82bb", alpha=0.85)
plt.plot(range(1, len(explained_var) + 1), np.cumsum(explained_var) * 100, marker="o", color="#d7191c")
plt.title("PCA Explained Variance Ratio (%)")
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained (%)")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# %% [markdown]
# ### 6. Decoupled Vulnerability & Heat Risk Index (HRI)
# We compute:
# $$\text{Risk} = \sqrt{\text{Exposure Score} \times \text{Vulnerability Score}}$$
# and perform unsupervised K-Means clustering into urban thermal-vulnerability archetypes.

# %%
df_risk, cluster_means = compute_vulnerability_and_risk(df_pred, region_key)

print("\n--- K-Means Urban Archetype Profiles ---")
print(cluster_means.to_string())

# %% [markdown]
# ### 7. Explainable AI with SHAP
# TreeExplainer is applied to discover non-linear factor tipping points and global drivers.

# %%
shap_results = run_shap_analysis(tree_model, df_risk, region_key, sample_size=800)
shap_importance = shap_results["importance_df"]

print("\n--- Top Drivers of Urban Heat ---")
print(shap_importance.head(8).to_string(index=False))

# %% [markdown]
# ### 8. Administrative Ward Aggregation & Municipal Decision Matrix
# Micro-grid cells are aggregated to official Municipal Wards, computing the **Ward Heat Risk Index (WHRI)**, priority ranks, and Heat Action Plan policies.

# %%
ward_stats = aggregate_to_administrative_wards(df_risk, region_key)

print("\n--- Top 10 Emergency Priority Wards Needing Government Intervention ---")
print(ward_stats[[
    "Priority_Rank", "ward_id", "ward_name", "Mean_LST_Celsius",
    "Ward_Heat_Risk_Index", "Risk_Category", "Recommended_Interventions"
]].head(10).to_string(index=False))
