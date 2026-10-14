"""
Master End-to-End Pipeline Execution Script.
Executes data generation, spatial ML modeling, SHAP explainability,
demographic vulnerability indexing, and municipal ward risk aggregation.
Usage:
    python scripts/run_pipeline.py --region trichy_delta
    python scripts/run_pipeline.py --region chennai_nondelta
    python scripts/run_pipeline.py --all
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import REGIONS
from src.data_generator import generate_synthetic_region_data
from src.models import train_and_evaluate_models
from src.vulnerability import compute_vulnerability_and_risk
from src.explainability import run_shap_analysis
from src.ward_aggregation import aggregate_to_administrative_wards

def run_pipeline_for_region(region_key="trichy_delta"):
    reg_info = REGIONS[region_key]
    print("\n" + "="*80)
    print(f"EXECUTING URBAN HEAT RISK PIPELINE FOR: {reg_info['name'].upper()}")
    print(f"Classification: {reg_info['type']}")
    print(f"Description: {reg_info['description']}")
    print("="*80)

    # Step 1: Data Generation / Harmonization
    print("\n[Step 1/5] Ingesting & harmonizing spatial grid and administrative boundaries...")
    df, wards_geojson = generate_synthetic_region_data(region_key)

    # Step 2: ML Model Training & Spatial CV
    print("\n[Step 2/5] Training supervised models and running Spatial Block Cross-Validation...")
    model_results = train_and_evaluate_models(df, region_key)
    best_model = model_results["best_model"]
    tree_model = model_results["tree_model"]
    best_model_name = model_results["best_model_name"]
    df_pred = model_results["df_with_predictions"]

    # Step 3: Vulnerability & Multi-Criteria Risk Indexing
    print("\n[Step 3/5] Decoupling Exposure vs Vulnerability and calculating Heat Risk Index...")
    df_risk, cluster_means = compute_vulnerability_and_risk(df_pred, region_key)

    # Step 4: Explainable AI with SHAP (Using Tree-based XGBoost)
    print("\n[Step 4/5] Computing SHAP values and generating non-linear factor threshold plots...")
    shap_results = run_shap_analysis(tree_model, df_risk, region_key)

    # Step 5: Ward Aggregation & Municipal Governance Policy Advisory
    print("\n[Step 5/5] Aggregating to Administrative Wards and generating Heat Action Plan priorities...")
    ward_stats = aggregate_to_administrative_wards(df_risk, region_key)

    print("\n" + "-"*80)
    print(f"TOP 5 EMERGENCY PRIORITY WARDS IN {reg_info['name'].upper()}:")
    print(ward_stats[["Priority_Rank", "ward_id", "ward_name", "Mean_LST_Celsius", "Ward_Heat_Risk_Index", "Risk_Category"]].head(5).to_string(index=False))
    print("-"*80)
    print(f"Pipeline finished successfully for {region_key}!\n")

def main():
    parser = argparse.ArgumentParser(description="Urban Heat Vulnerability ML Pipeline")
    parser.add_argument("--region", choices=["trichy_delta", "chennai_nondelta"], default="trichy_delta")
    parser.add_argument("--all", action="store_true", help="Run both Delta and Non-Delta regions")
    args = parser.parse_args()

    if args.all:
        for reg in ["trichy_delta", "chennai_nondelta"]:
            run_pipeline_for_region(reg)
    else:
        run_pipeline_for_region(args.region)

if __name__ == "__main__":
    main()
