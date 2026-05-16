"""
src/data_generator.py
---------------------
Generates a realistic synthetic Indian Liver Patient Dataset.

Features are based on the original ILPD (Indian Liver Patient Dataset)
published by UCI ML Repository — Ramana et al., 2012.

All biological distributions reflect published clinical ranges for
Indian patients with and without liver disease:
  - Total Bilirubin       (µmol/L)
  - Direct Bilirubin      (µmol/L)
  - Alkaline Phosphotase  (IU/L)
  - Alamine Aminotransferase / SGPT  (IU/L)
  - Aspartate Aminotransferase / SGOT (IU/L)
  - Total Proteins        (g/dL)
  - Albumin               (g/dL)
  - Albumin/Globulin Ratio
  - Age, Gender

Target: 1 = Liver Disease  ·  2 = No Liver Disease
(matches original ILPD encoding)
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG  = np.random.default_rng(42)
N    = 583   # matches original ILPD size


def generate_dataset(n: int = N, save_path=None) -> pd.DataFrame:
    """
    Generate n synthetic Indian liver patient records.
    Class split: ~71% disease (415), ~29% no disease (168) — mirrors original ILPD.
    """
    n_disease = int(n * 0.712)
    n_healthy = n - n_disease

    rows = []

    for label, n_grp in [(1, n_disease), (2, n_healthy)]:
        is_disease = (label == 1)

        # Age: Indian clinical cohort, 4–90
        age = RNG.integers(4, 90, n_grp).astype(float)

        # Gender: ~75% male in ILPD (liver disease more common in males)
        gender_prob = 0.75 if is_disease else 0.60
        gender = RNG.choice(["Male", "Female"], n_grp,
                            p=[gender_prob, 1-gender_prob])

        # Total Bilirubin (mg/dL)
        # Healthy: 0.1–1.2   Disease: 0.5–30+
        if is_disease:
            tb = np.abs(RNG.lognormal(mean=1.2, sigma=1.0, size=n_grp)).clip(0.4, 75)
        else:
            tb = np.abs(RNG.lognormal(mean=0.0, sigma=0.35, size=n_grp)).clip(0.1, 1.2)

        # Direct Bilirubin: correlated with total (~40-60%)
        db = (tb * RNG.uniform(0.30, 0.65, n_grp)).clip(0.0, 20)

        # Alkaline Phosphotase (IU/L)
        # Healthy: 44–147   Disease: 60–2110 (can be very high in cholestasis)
        if is_disease:
            alkphos = np.abs(RNG.lognormal(mean=5.3, sigma=0.9, size=n_grp)).clip(60, 2110)
        else:
            alkphos = np.abs(RNG.lognormal(mean=4.7, sigma=0.25, size=n_grp)).clip(44, 160)

        # SGPT / ALT (IU/L) — hepatocellular damage marker
        if is_disease:
            sgpt = np.abs(RNG.lognormal(mean=4.0, sigma=1.0, size=n_grp)).clip(10, 2000)
        else:
            sgpt = np.abs(RNG.lognormal(mean=2.8, sigma=0.45, size=n_grp)).clip(7, 56)

        # SGOT / AST (IU/L) — correlated with SGPT
        sgot_ratio = RNG.uniform(0.7, 2.5, n_grp) if is_disease else RNG.uniform(0.6, 1.2, n_grp)
        sgot = (sgpt * sgot_ratio).clip(10, 4929)

        # Total Proteins (g/dL): 6–8.3 healthy, can drop in disease
        if is_disease:
            tp = RNG.normal(loc=6.4, scale=1.0, size=n_grp).clip(2.7, 9.6)
        else:
            tp = RNG.normal(loc=7.0, scale=0.55, size=n_grp).clip(5.5, 9.0)

        # Albumin (g/dL): drops in liver failure
        if is_disease:
            alb = RNG.normal(loc=3.0, scale=0.8, size=n_grp).clip(0.9, 5.5)
        else:
            alb = RNG.normal(loc=3.8, scale=0.45, size=n_grp).clip(2.8, 5.5)

        # Albumin/Globulin Ratio: Albumin / (Total Protein – Albumin)
        globulin = (tp - alb).clip(0.1, 6)
        ag_ratio = (alb / globulin).clip(0.3, 2.8)
        # Small fraction of AG ratio is missing in original (add ~5% NaN)
        ag_ratio_with_nan = ag_ratio.astype(object)
        nan_idx = RNG.choice(n_grp, size=max(1, int(n_grp * 0.05)), replace=False)
        ag_ratio_with_nan[nan_idx] = np.nan

        for i in range(n_grp):
            rows.append({
                "Age":                    int(age[i]),
                "Gender":                 gender[i],
                "Total_Bilirubin":        round(float(tb[i]), 1),
                "Direct_Bilirubin":       round(float(db[i]), 1),
                "Alkaline_Phosphotase":   int(alkphos[i]),
                "Alamine_Aminotransferase": int(sgpt[i]),
                "Aspartate_Aminotransferase": int(sgot[i]),
                "Total_Protiens":         round(float(tp[i]), 1),
                "Albumin":                round(float(alb[i]), 1),
                "Albumin_and_Globulin_Ratio": float(ag_ratio_with_nan[i]) if ag_ratio_with_nan[i] is not np.nan else np.nan,
                "Dataset":                label,   # 1=Disease 2=No Disease
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)

    return df


# Feature metadata
FEATURES = [
    "Age", "Gender",
    "Total_Bilirubin", "Direct_Bilirubin",
    "Alkaline_Phosphotase", "Alamine_Aminotransferase", "Aspartate_Aminotransferase",
    "Total_Protiens", "Albumin", "Albumin_and_Globulin_Ratio",
]
NUMERIC_FEATURES = [f for f in FEATURES if f != "Gender"]

FEATURE_LABELS = {
    "Age":                           "Age (years)",
    "Gender":                        "Gender",
    "Total_Bilirubin":               "Total Bilirubin (mg/dL)",
    "Direct_Bilirubin":              "Direct Bilirubin (mg/dL)",
    "Alkaline_Phosphotase":          "Alkaline Phosphotase (IU/L)",
    "Alamine_Aminotransferase":      "SGPT / ALT (IU/L)",
    "Aspartate_Aminotransferase":    "SGOT / AST (IU/L)",
    "Total_Protiens":                "Total Proteins (g/dL)",
    "Albumin":                       "Albumin (g/dL)",
    "Albumin_and_Globulin_Ratio":    "Albumin/Globulin Ratio",
}

CLASS_LABELS  = {1: "Liver Disease", 2: "No Disease"}
CLASS_COLORS  = {1: "#EF4444", 2: "#22C55E"}


if __name__ == "__main__":
    df = generate_dataset(save_path="data/indian_liver_patients.csv")
    print(f"Generated {len(df)} records")
    print(f"Class distribution:\n{df['Dataset'].value_counts().sort_index()}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum()>0]}")
    print(df.describe().round(2))
