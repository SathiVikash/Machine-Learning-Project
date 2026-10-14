"""
Scientific Scenario Service for Municipal Heat Interventions.
Translates planned urban interventions (cool roofs, tree canopy expansion, de-paving,
water body restoration) into feature modifications, strictly clamps them
within the model's domain of empirical validity, and evaluates ML model predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from src.config import (
    FEATURE_PHYSICAL_BOUNDS,
    EXPOSURE_FEATURES,
    TARGET_VARIABLE
)

class ScenarioService:
    def __init__(self):
        self.bounds = FEATURE_PHYSICAL_BOUNDS

    def clamp_feature(self, feature_name: str, value: float) -> Tuple[float, bool]:
        """
        Clamps a feature strictly within physical and empirical model boundaries.
        Returns (clamped_value, was_clamped).
        """
        if feature_name not in self.bounds:
            return value, False
        
        b = self.bounds[feature_name]
        min_val = b["min"]
        max_val = b["max"]

        if value < min_val:
            return min_val, True
        elif value > max_val:
            return max_val, True
        return value, False

    def apply_preset_intervention(self, base_features: Dict[str, Any], intervention_type: str, intensity: float = 1.0) -> Dict[str, Any]:
        """
        Applies a validated municipal intervention package with an intensity factor (0.5 to 1.5).
        Returns a new feature dictionary with applied shifts.
        """
        scenario = dict(base_features)

        if intervention_type == "cool_roof":
            # Cool roofs reflect shortwave solar radiation and replace dark impervious bitumen
            # Albedo increases (+0.08 to +0.20), NDBI decreases (-0.05 to -0.15)
            scenario["Albedo"] = scenario.get("Albedo", 0.18) + (0.14 * intensity)
            scenario["NDBI"] = scenario.get("NDBI", 0.15) - (0.10 * intensity)

        elif intervention_type == "urban_greening":
            # Tree canopy expansion increases NDVI (+0.10 to +0.25), decreases NDBI, reduces park distance
            scenario["NDVI"] = scenario.get("NDVI", 0.20) + (0.16 * intensity)
            scenario["NDBI"] = scenario.get("NDBI", 0.15) - (0.07 * intensity)
            current_dist_park = scenario.get("Dist_to_Park_m", 1500)
            scenario["Dist_to_Park_m"] = max(100.0, current_dist_park * (1.0 - 0.5 * intensity))

        elif intervention_type == "de_paving":
            # De-paving concrete/asphalt and replacing with permeable soil/grass
            scenario["Building_Density_pct"] = max(5.0, scenario.get("Building_Density_pct", 50.0) * (1.0 - 0.20 * intensity))
            scenario["NDVI"] = scenario.get("NDVI", 0.20) + (0.08 * intensity)
            scenario["NDBI"] = scenario.get("NDBI", 0.15) - (0.09 * intensity)

        elif intervention_type == "water_corridor":
            # Urban canal rejuvenation / retention pond installation
            scenario["Dist_to_Water_m"] = min(scenario.get("Dist_to_Water_m", 2000), max(50.0, 300.0 / intensity))
            scenario["NDWI"] = scenario.get("NDWI", -0.10) + (0.12 * intensity)

        elif intervention_type == "comprehensive_hap":
            # Combined municipal resilience package: cool roofs + street trees + pocket greening
            scenario["Albedo"] = scenario.get("Albedo", 0.18) + (0.12 * intensity)
            scenario["NDVI"] = scenario.get("NDVI", 0.20) + (0.14 * intensity)
            scenario["NDBI"] = scenario.get("NDBI", 0.15) - (0.12 * intensity)
            scenario["Building_Density_pct"] = max(5.0, scenario.get("Building_Density_pct", 50.0) * (1.0 - 0.12 * intensity))
            scenario["Dist_to_Park_m"] = max(150.0, scenario.get("Dist_to_Park_m", 1500) * 0.5)

        return scenario

    def evaluate_scenario(
        self,
        model,
        baseline_row: Dict[str, Any],
        scenario_features: Dict[str, Any],
        vulnerability_score: float = 0.5
    ) -> Dict[str, Any]:
        """
        Evaluates the ML model on baseline vs scenario inputs.
        Enforces domain clamping, calculates delta LST, recalculates HRI,
        and logs assumptions and limitations.
        """
        # Ensure all required exposure features exist
        base_vec = {}
        scen_vec = {}
        clamped_log = []

        for f in EXPOSURE_FEATURES:
            # Baseline value
            b_val = float(baseline_row.get(f, 0.0))
            base_vec[f] = b_val

            # Scenario value with clamping
            s_val = float(scenario_features.get(f, b_val))
            clamped_s_val, was_clamped = self.clamp_feature(f, s_val)
            scen_vec[f] = clamped_s_val

            if was_clamped:
                clamped_log.append({
                    "feature": f,
                    "requested": s_val,
                    "clamped_to": clamped_s_val,
                    "reason": f"Input fell outside empirical bounds [{self.bounds[f]['min']}, {self.bounds[f]['max']}]"
                })

        # Convert to DataFrames matching model feature order
        df_base = pd.DataFrame([base_vec])[EXPOSURE_FEATURES]
        df_scen = pd.DataFrame([scen_vec])[EXPOSURE_FEATURES]

        # Model Inference
        pred_base_lst = float(model.predict(df_base)[0])
        pred_scen_lst = float(model.predict(df_scen)[0])
        delta_lst = pred_scen_lst - pred_base_lst  # Negative means cooling

        # Exposure Score Normalization (approx empirical scale: 28°C = 0.0, 48°C = 1.0)
        exp_base = np.clip((pred_base_lst - 28.0) / (48.0 - 28.0), 0.0, 1.0)
        exp_scen = np.clip((pred_scen_lst - 28.0) / (48.0 - 28.0), 0.0, 1.0)

        # HRI = sqrt(Exposure * Vulnerability)
        v_clipped = np.clip(vulnerability_score, 0.0, 1.0)
        hri_base = float(np.sqrt(exp_base * v_clipped))
        hri_scen = float(np.sqrt(exp_scen * v_clipped))
        delta_hri = hri_scen - hri_base

        # Scientific Assumptions & Model Limitations
        assumptions = [
            "Random Forest regression models empirical surface temperature (LST) correlations, not micro-climate CFD dynamics.",
            "Predictions assume cloudless midday solar conditions consistent with summer training imagery.",
            "Vegetative cooling benefits (NDVI) require adequate soil moisture and canopy establishment.",
            "SHAP and tree ensemble responses reflect statistical association within regional training bounds."
        ]

        return {
            "baseline": {
                "features": base_vec,
                "predicted_lst_c": round(pred_base_lst, 2),
                "exposure_score": round(float(exp_base), 3),
                "heat_risk_index": round(hri_base, 3)
            },
            "scenario": {
                "features": scen_vec,
                "predicted_lst_c": round(pred_scen_lst, 2),
                "exposure_score": round(float(exp_scen), 3),
                "heat_risk_index": round(hri_scen, 3)
            },
            "delta": {
                "cooling_celsius": round(-delta_lst, 2),  # Positive value indicates cooling
                "delta_lst_raw": round(delta_lst, 2),
                "hri_reduction": round(-delta_hri, 3),
                "pct_hri_improvement": round((-delta_hri / (hri_base + 1e-6)) * 100.0, 1)
            },
            "clamped_features": clamped_log,
            "assumptions_and_limitations": assumptions,
            "is_valid_domain": len(clamped_log) == 0
        }

# Global instance
scenario_service = ScenarioService()
