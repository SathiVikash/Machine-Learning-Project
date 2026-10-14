"""
Machine Learning Modeling and Evaluation Module.
Implements:
1. Linear Regression (Baseline)
2. Random Forest Regressor
3. XGBoost Regressor
4. PCA Dimensionality Reduction
5. Spatial Block Cross-Validation & Metric Comparison (R², RMSE, MAE)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import xgboost as xgb
import joblib

from src.config import (
    EXPOSURE_FEATURES,
    TARGET_VARIABLE,
    MODELS_DIR,
    REGIONS
)

class SpatialBlockKFold:
    """
    Spatial K-Fold cross-validator to prevent spatial autocorrelation leakage.
    Splits spatial observations into geographic grid blocks.
    """
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, df):
        np.random.seed(self.random_state)
        # Partition space into spatial blocks based on lat/lon
        n_blocks = int(np.ceil(np.sqrt(self.n_splits)))
        lat_bins = pd.qcut(df["lat"], q=n_blocks, labels=False, duplicates="drop")
        lon_bins = pd.qcut(df["lon"], q=n_blocks, labels=False, duplicates="drop")
        block_id = lat_bins * 10 + lon_bins
        unique_blocks = np.unique(block_id)
        
        # Assign blocks to folds
        block_fold_map = {b: i % self.n_splits for i, b in enumerate(np.random.permutation(unique_blocks))}
        folds = block_id.map(block_fold_map).values
        
        for fold in range(self.n_splits):
            val_idx = np.where(folds == fold)[0]
            train_idx = np.where(folds != fold)[0]
            yield train_idx, val_idx

def train_and_evaluate_models(df, region_key="trichy_delta"):
    """
    Trains and compares Linear Regression, Random Forest, and XGBoost using Spatial CV.
    Returns trained models, evaluation metrics summary DataFrame, and feature scaler.
    """
    X = df[EXPOSURE_FEATURES].copy()
    y = df[TARGET_VARIABLE].values

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=EXPOSURE_FEATURES)

    # 1. PCA Analysis for Feature Redundancy
    pca = PCA(n_components=min(5, len(EXPOSURE_FEATURES)))
    pca.fit(X_scaled)
    explained_var = pca.explained_variance_ratio_

    # 2. Model Definitions
    models = {
        "Ridge Regression (Baseline, L2 alpha=1.0)": Ridge(alpha=1.0),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=150, max_depth=10, min_samples_split=4, random_state=42, n_jobs=-1
        ),
        "XGBoost Regressor": xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.08, subsample=0.85,
            colsample_bytree=0.85, random_state=42, n_jobs=-1
        )
    }

    # 3. Spatial Cross-Validation
    spatial_cv = SpatialBlockKFold(n_splits=5)
    results = []

    for name, model in models.items():
        r2_scores = []
        rmse_scores = []
        mae_scores = []
        
        for train_idx, val_idx in spatial_cv.split(df):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            if "Ridge" in name or "Linear" in name:
                # Scale for linear/ridge regression
                X_tr_sc = scaler.transform(X_train)
                X_vl_sc = scaler.transform(X_val)
                model.fit(X_tr_sc, y_train)
                preds = model.predict(X_vl_sc)
            else:
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                
            r2_scores.append(r2_score(y_val, preds))
            rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))
            mae_scores.append(mean_absolute_error(y_val, preds))
            
        results.append({
            "Model": name,
            "Spatial_CV_R2_Mean": np.round(np.mean(r2_scores), 4),
            "Spatial_CV_R2_Std": np.round(np.std(r2_scores), 4),
            "Spatial_CV_RMSE_degC": np.round(np.mean(rmse_scores), 3),
            "Spatial_CV_MAE_degC": np.round(np.mean(mae_scores), 3)
        })

    metrics_df = pd.DataFrame(results).sort_values(by="Spatial_CV_R2_Mean", ascending=False)
    
    # 4. Save trained tree model (Random Forest) specifically for SHAP non-linear discovery
    rf_model = models["Random Forest Regressor"]
    rf_model.fit(X, y)
    rf_save_path = MODELS_DIR / f"{region_key}_rf_model.joblib"
    joblib.dump(rf_model, rf_save_path)

    # Fit Best Model on Full Dataset
    best_model_name = metrics_df.iloc[0]["Model"]
    best_model = models[best_model_name]
    if "Linear" in best_model_name:
        best_model.fit(scaler.transform(X), y)
    else:
        best_model.fit(X, y)
    
    # Generate predicted LST and Exposure Index using best model
    if "Linear" in best_model_name:
        df["Predicted_LST"] = np.round(best_model.predict(scaler.transform(X)), 2)
    else:
        df["Predicted_LST"] = np.round(best_model.predict(X), 2)
    df["Prediction_Residual"] = np.round(df["LST_Celsius"] - df["Predicted_LST"], 2)
    
    # Save artifacts
    model_save_path = MODELS_DIR / f"{region_key}_best_model_{best_model_name.replace(' ', '_')}.joblib"
    scaler_save_path = MODELS_DIR / f"{region_key}_scaler.joblib"
    joblib.dump(best_model, model_save_path)
    joblib.dump(scaler, scaler_save_path)

    metrics_save_path = MODELS_DIR / f"{region_key}_model_comparison.csv"
    metrics_df.to_csv(metrics_save_path, index=False)

    print(f"[{region_key}] Model Training Completed.")
    print(metrics_df.to_string(index=False))
    print(f"PCA Top-3 Components Variance Explained: {np.round(explained_var[:3] * 100, 2)}%")
    print(f"Saved best model ({best_model_name}) -> {model_save_path}")
    print(f"Saved Random Forest tree model for SHAP -> {rf_save_path}")

    return {
        "best_model": best_model,
        "best_model_name": best_model_name,
        "tree_model": rf_model,
        "metrics_df": metrics_df,
        "scaler": scaler,
        "pca_variance": explained_var,
        "df_with_predictions": df
    }
