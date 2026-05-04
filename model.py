"""
NutriAI – Enhanced Diet Recommendation Engine
================================================
Features
---------
• KNN-based food similarity (10-feature space)
• Health-risk classification (Random Forest on patient dataset)
• Disease-specific dietary rules (Diabetes, Hypertension, Obesity, Heart Disease)
• TDEE / BMI / Macros / Body-composition
• Micronutrient optimisation
• Food menu: browse, select, get safety/suitability rating per food per user
• Meal plan generator (single-day + 7-day weekly, no repetition)
• Hydration calculator
• Diet duration & phased roadmap
• Health insights & disease warnings
• Foods merged from calorie_calculator + foods_data CSVs
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import silhouette_score, classification_report
from sklearn.cluster import KMeans
from collections import deque


# ═══════════════════════════════════════════════════════════
#  DISEASE RULES  (applied on top of baseline macros)
# ═══════════════════════════════════════════════════════════
DISEASE_RULES = {
    "Diabetes": {
        "protocol":    "Low-GI, High-Fiber, Controlled Carbohydrate",
        "carb_pct":    0.35,   # reduce from 0.40
        "protein_pct": 0.30,
        "fat_pct":     0.35,
        "max_sodium":  2300,
        "max_sugar_gi": "Low",
        "avoid_tags":  ["High GI", "high_sugar"],
        "prefer_tags": ["diabetes_friendly", "High Fiber", "Low GI"],
        "calorie_adj": -200,   # moderate deficit
        "notes": [
            "Limit refined carbs and sugary foods (GI > 70).",
            "Spread meals across 5–6 small portions to stabilise glucose.",
            "Prioritise legumes, whole grains, non-starchy vegetables.",
            "Monitor glucose 2 hrs post-meal; target <140 mg/dL.",
            "Avoid fruit juices; prefer whole fruit with skin."
        ]
    },
    "Hypertension": {
        "protocol":    "DASH – Low Sodium, High Potassium",
        "carb_pct":    0.40,
        "protein_pct": 0.25,
        "fat_pct":     0.35,
        "max_sodium":  1500,   # strict DASH target
        "prefer_tags": ["bp_friendly", "Low Sodium"],
        "avoid_tags":  ["high_sodium", "processed"],
        "calorie_adj": -150,
        "notes": [
            "Restrict sodium to <1500 mg/day (DASH protocol).",
            "Increase potassium-rich foods: bananas, spinach, sweet potatoes.",
            "Limit saturated fat; choose olive oil, avocado, nuts.",
            "Avoid pickled, cured, canned, and fast foods.",
            "Limit caffeine – monitor BP after coffee/tea.",
            "Reduce alcohol: max 1 drink/day (women), 2 (men)."
        ]
    },
    "Obesity": {
        "protocol":    "High-Protein, High-Fiber, Calorie-Deficit",
        "carb_pct":    0.30,
        "protein_pct": 0.35,
        "fat_pct":     0.35,
        "max_sodium":  2300,
        "prefer_tags": ["Low Calorie", "High Fiber", "High Protein"],
        "avoid_tags":  ["high_fat", "fried", "processed"],
        "calorie_adj": -600,   # aggressive but safe deficit
        "notes": [
            "Target 0.5–0.75 kg/week fat loss – never exceed 1 kg/week.",
            "Increase protein to 1.6–2.0 g/kg bodyweight to preserve muscle.",
            "Eat slowly; satiety signals take 20 minutes to register.",
            "Fill half the plate with non-starchy vegetables.",
            "Avoid liquid calories (juice, soda, alcohol).",
            "Resistance training 3×/week minimises muscle loss during deficit."
        ]
    },
    "Heart Disease": {
        "protocol":    "Heart-Healthy – Low Saturated Fat, Low Sodium, High Omega-3",
        "carb_pct":    0.45,
        "protein_pct": 0.25,
        "fat_pct":     0.30,
        "max_sodium":  1500,
        "prefer_tags": ["Heart Healthy", "Low Fat", "bp_friendly"],
        "avoid_tags":  ["high_fat", "high_sodium", "trans_fat"],
        "calorie_adj": -200,
        "notes": [
            "Replace saturated fats with mono/polyunsaturated (olive oil, nuts, fish).",
            "Eat fatty fish (salmon, mackerel) 2× per week for Omega-3.",
            "Target dietary cholesterol <200 mg/day.",
            "Increase soluble fiber (oats, beans, lentils) to lower LDL.",
            "Limit sodium to 1500 mg/day to reduce cardiac load.",
            "Avoid trans fats completely – check labels for 'partially hydrogenated'."
        ]
    }
}

SEVERITY_MULTIPLIERS = {
    "Mild":     1.0,
    "Moderate": 1.15,  # stricter restrictions
    "Severe":   1.30
}


# ═══════════════════════════════════════════════════════════
#  MAIN ENGINE
# ═══════════════════════════════════════════════════════════
class DietRecommendationEngine:

    DIET_HIERARCHY = {
        "Vegan":       ["Vegan"],
        "Vegetarian":  ["Vegetarian", "Vegan"],
        "Pescatarian": ["Pescatarian", "Vegetarian", "Vegan"],
        "Non-Veg":     ["Non-Veg", "Vegetarian", "Vegan"],
        "All":         ["Non-Veg", "Vegetarian", "Vegan", "Pescatarian"],
    }

    def __init__(self,
                 data_path: str = "foods_data.csv",
                 calorie_data_path: str = "calorie_calculator_diet_dataset.csv",
                 patient_data_path: str = "diet_recommendations_dataset.csv"):

        self.data_path         = data_path
        self.calorie_data_path = calorie_data_path
        self.patient_data_path = patient_data_path

        self._load_and_merge_foods()
        self._preprocess_data()
        self._train_knn()
        self._train_health_risk_classifier()
        self._evaluate_model()

    # ─────────────────────────────────────────────
    # DATA LOADING & MERGING
    # ─────────────────────────────────────────────
    def _load_and_merge_foods(self):
        """Merge foods_data.csv and calorie_calculator_diet_dataset.csv into unified food DB."""
        import os

        frames = []

        # --- Primary foods_data ---
        if os.path.exists(self.data_path):
            df1 = pd.read_csv(self.data_path)
            # Standardise column names
            rename1 = {"name": "name"}
            df1 = df1.rename(columns=rename1)
            required = ["calories", "protein", "carbs", "fat"]
            for c in required:
                if c in df1.columns:
                    df1[c] = pd.to_numeric(df1[c], errors="coerce")
            frames.append(df1)
            print(f"✅ Loaded foods_data.csv: {len(df1)} items")

        # --- Calorie calculator dataset ---
        if os.path.exists(self.calorie_data_path):
            df2 = pd.read_csv(self.calorie_data_path)
            # Map columns to unified schema
            df2 = df2.rename(columns={
                "food_name":    "name",
                "protein_g":    "protein",
                "carbs_g":      "carbs",
                "fat_g":        "fat",
                "fiber_g":      "fiber",
                "origin":       "country",
            })
            # Preserve disease-friendly flags
            df2["diabetes_friendly_flag"] = (df2.get("diabetes_friendly", "No") == "Yes").astype(int)
            df2["bp_friendly_flag"]        = (df2.get("bp_friendly",      "No") == "Yes").astype(int)
            df2["glycemic_index_val"]      = df2.get("glycemic_index", "Medium")
            df2["meal_type_tag"]           = df2.get("meal_type", "")
            df2["health_goal_tag"]         = df2.get("health_goal", "")
            df2["allergens_tag"]           = df2.get("allergens", "")
            df2["prep_difficulty_tag"]     = df2.get("prep_difficulty", "")

            if "budget" not in df2.columns:
                df2["budget"] = "Low"
            if "season" not in df2.columns:
                df2["season"] = "All"

            frames.append(df2)
            print(f"✅ Loaded calorie_calculator.csv: {len(df2)} items")

        if not frames:
            raise FileNotFoundError("No food data files found!")

        self.df = pd.concat(frames, ignore_index=True)

        # De-duplicate by name (keep first occurrence)
        self.df["_name_lower"] = self.df["name"].str.strip().str.lower()
        self.df = self.df.drop_duplicates(subset=["_name_lower"]).drop(columns=["_name_lower"])
        self.df = self.df.reset_index(drop=True)
        print(f"✅ Merged food database: {len(self.df)} unique items")

    # ─────────────────────────────────────────────
    # PREPROCESSING
    # ─────────────────────────────────────────────
    def _preprocess_data(self):
        df = self.df

        for col in ["calories", "protein", "fat", "carbs"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df = df[df["calories"] > 0].reset_index(drop=True)

        for col in ["fiber", "sodium_mg"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            else:
                df[col] = 0

        for col in ["diet_type", "country", "season", "budget", "micronutrients"]:
            if col not in df.columns:
                df[col] = "All" if col in ("country", "season", "budget") else ""
            df[col] = df[col].fillna("All" if col in ("country", "season", "budget") else "")

        # Disease flags (from calorie dataset)
        for flag in ["diabetes_friendly_flag", "bp_friendly_flag"]:
            if flag not in df.columns:
                df[flag] = 0
            else:
                df[flag] = pd.to_numeric(df[flag], errors="coerce").fillna(0)

        if "glycemic_index_val" not in df.columns:
            df["glycemic_index_val"] = "Medium"
        else:
            df["glycemic_index_val"] = df["glycemic_index_val"].fillna("Medium")

        # GI numeric mapping
        gi_map = {"Low": 1, "Medium": 2, "High": 3}
        df["gi_score"] = df["glycemic_index_val"].map(gi_map).fillna(2)

        # Tags
        for tag_col in ["meal_type_tag", "health_goal_tag", "allergens_tag"]:
            if tag_col not in df.columns:
                df[tag_col] = ""
            df[tag_col] = df[tag_col].fillna("")

        # Derived nutritional features
        cal = df["calories"].clip(lower=1)
        df["protein_pct"]      = (df["protein"] * 4   / cal) * 100
        df["carb_pct"]         = (df["carbs"]   * 4   / cal) * 100
        df["fat_pct"]          = (df["fat"]      * 9   / cal) * 100
        df["nutrient_density"] = (
            df["protein"] * 4.0
            - df["fat"]   * 0.5
            + df["carbs"] * 0.2
            + df["fiber"] * 2.0
        ) / cal
        df["fiber_density"] = df["fiber"] / cal * 100
        df["sodium_density"] = df["sodium_mg"] / cal.clip(lower=1)

        # Health-safety composite score (higher = healthier)
        df["health_safety_score"] = (
            df["diabetes_friendly_flag"] * 10
            + df["bp_friendly_flag"]     * 10
            + (3 - df["gi_score"])       * 5   # low GI gets +10
            + df["fiber"]                * 2
            - df["sodium_mg"]            / 500 * 5
            + df["protein"]              * 1
        )

        self.df = df

        self.features = [
            "calories", "protein", "fat", "carbs",
            "protein_pct", "carb_pct", "fat_pct",
            "nutrient_density", "fiber", "fiber_density",
            "sodium_density", "gi_score", "health_safety_score"
        ]

        self.scaler = StandardScaler()
        self.X_all  = self.scaler.fit_transform(self.df[self.features])
        print(f"✅ Preprocessing done. Features: {len(self.features)}, Items: {len(self.df)}")

    # ─────────────────────────────────────────────
    # KNN TRAINING
    # ─────────────────────────────────────────────
    def _train_knn(self):
        k = min(20, len(self.X_all) - 1)
        self.knn_model = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="euclidean")
        self.knn_model.fit(self.X_all)
        print("✅ KNN model trained (13-feature space)")

    # ─────────────────────────────────────────────
    # HEALTH-RISK CLASSIFIER  (Random Forest on patient dataset)
    # ─────────────────────────────────────────────
    def _train_health_risk_classifier(self):
        import os
        if not os.path.exists(self.patient_data_path):
            self.risk_clf = None
            self.risk_encoder = None
            print("⚠️ Patient dataset not found – health-risk classifier disabled.")
            return

        pat = pd.read_csv(self.patient_data_path)

        # Feature engineering
        pat["BMI"]            = pd.to_numeric(pat["BMI"], errors="coerce")
        pat["Age"]            = pd.to_numeric(pat["Age"], errors="coerce")
        pat["Glucose_mg/dL"]  = pd.to_numeric(pat["Glucose_mg/dL"], errors="coerce")
        pat["Cholesterol_mg/dL"] = pd.to_numeric(pat["Cholesterol_mg/dL"], errors="coerce")

        # Parse systolic BP from "120/80" format
        def parse_systolic(bp_str):
            try:
                return int(str(bp_str).split("/")[0])
            except Exception:
                return 120
        pat["systolic_bp"] = pat["Blood_Pressure_mmHg"].apply(parse_systolic)

        le_gender   = LabelEncoder()
        le_activity = LabelEncoder()
        le_severity = LabelEncoder()
        pat["gender_enc"]   = le_gender.fit_transform(pat["Gender"].fillna("Male"))
        pat["activity_enc"] = le_activity.fit_transform(pat["Physical_Activity_Level"].fillna("Moderate"))
        pat["severity_enc"] = le_severity.fit_transform(pat["Severity"].fillna("Mild"))

        feature_cols = [
            "Age", "BMI", "Glucose_mg/dL", "Cholesterol_mg/dL",
            "systolic_bp", "Daily_Caloric_Intake",
            "Weekly_Exercise_Hours", "Dietary_Nutrient_Imbalance_Score",
            "gender_enc", "activity_enc", "severity_enc"
        ]

        X = pat[feature_cols].fillna(0)
        y = pat["Diet_Recommendation"].fillna("Balanced")

        self.risk_label_encoder = LabelEncoder()
        y_enc = self.risk_label_encoder.fit_transform(y)

        self.risk_feature_cols = feature_cols
        self.risk_scaler       = StandardScaler()
        X_scaled               = self.risk_scaler.fit_transform(X)

        self.risk_clf = RandomForestClassifier(
            n_estimators=200, max_depth=8,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        self.risk_clf.fit(X_scaled, y_enc)

        # Store encoders for inference
        self.risk_gender_le   = le_gender
        self.risk_activity_le = le_activity
        self.risk_severity_le = le_severity

        # CV score
        cv_scores = cross_val_score(
            RandomForestClassifier(n_estimators=100, random_state=42),
            X_scaled, y_enc, cv=5, scoring="accuracy"
        )
        self.risk_clf_accuracy = float(np.mean(cv_scores))
        print(f"✅ Health-risk classifier trained. CV Accuracy: {self.risk_clf_accuracy:.3f}")

    # ─────────────────────────────────────────────
    # EVALUATION
    # ─────────────────────────────────────────────
    def _evaluate_model(self):
        print("\n📊 MODEL EVALUATION")
        print("=" * 45)

        X_cv = self.df[self.features].values
        y_cv = self.df["calories"].values
        rf   = RandomForestRegressor(n_estimators=100, random_state=42)
        cv   = KFold(n_splits=5, shuffle=True, random_state=42)
        mse  = -cross_val_score(rf, X_cv, y_cv, cv=cv, scoring="neg_mean_squared_error")
        rmse = float(np.sqrt(np.mean(mse)))

        n_eval = min(50, len(self.df))
        idxs   = np.random.RandomState(42).choice(len(self.df), n_eval, replace=False)
        prec_scores = []
        for i in idxs:
            distances, indices = self.knn_model.kneighbors([self.X_all[i]], n_neighbors=6)
            neighbors  = [j for j in indices[0] if j != i][:5]
            target_cal = self.df.iloc[i]["calories"]
            relevant   = sum(1 for j in neighbors
                             if abs(self.df.iloc[j]["calories"] - target_cal) < 80)
            prec_scores.append(relevant / 5)
        precision_at_k = float(np.mean(prec_scores))

        n_clusters = min(8, len(self.df) // 5)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.X_all)
        sil    = float(silhouette_score(self.X_all, labels))

        risk_acc = getattr(self, "risk_clf_accuracy", None)
        print(f"🔹 KNN Feature Space:     {len(self.features)} features")
        print(f"🔹 Total Foods:           {len(self.df)}")
        print(f"🔹 RMSE (5-fold CV):      {rmse:.2f} kcal")
        print(f"🔹 Precision@5:           {precision_at_k:.4f}")
        print(f"🔹 Silhouette Score:      {sil:.4f}")
        if risk_acc:
            print(f"🔹 Health-Risk Accuracy:  {risk_acc:.4f}")
        print("=" * 45)

        self.metrics = {
            "rmse":                round(rmse, 2),
            "precision_at_k":      round(precision_at_k, 4),
            "silhouette_score":    round(sil, 4),
            "health_risk_accuracy": round(risk_acc, 4) if risk_acc else None,
            "total_foods":         len(self.df),
            "feature_count":       len(self.features),
        }

    # ═══════════════════════════════════════════════════════
    #  PUBLIC API
    # ═══════════════════════════════════════════════════════

    # ─────────────────────────────────────────────
    # HEALTH-RISK PREDICTION
    # ─────────────────────────────────────────────
    def predict_health_risk(self, age, weight_kg, height_cm, gender,
                             activity, glucose=100, cholesterol=180,
                             systolic_bp=120, caloric_intake=2000,
                             exercise_hours=3, nutrient_imbalance=2,
                             severity="Mild"):
        """Predict recommended diet protocol from patient health indicators."""
        if self.risk_clf is None:
            return {"protocol": "Balanced", "confidence": 0.0, "probabilities": {}}

        def safe_transform(encoder, val, default=0):
            try:
                return int(encoder.transform([val])[0])
            except Exception:
                return default

        bmi = round(weight_kg / (height_cm / 100) ** 2, 1)
        gender_enc   = safe_transform(self.risk_gender_le,   gender.capitalize())
        activity_enc = safe_transform(self.risk_activity_le, activity)
        severity_enc = safe_transform(self.risk_severity_le, severity)

        X = np.array([[
            age, bmi, glucose, cholesterol, systolic_bp,
            caloric_intake, exercise_hours, nutrient_imbalance,
            gender_enc, activity_enc, severity_enc
        ]])
        X_scaled = self.risk_scaler.transform(X)

        y_pred  = self.risk_clf.predict(X_scaled)[0]
        y_proba = self.risk_clf.predict_proba(X_scaled)[0]

        label       = self.risk_label_encoder.inverse_transform([y_pred])[0]
        classes     = self.risk_label_encoder.classes_
        probability = float(np.max(y_proba))

        proba_dict = {c: round(float(p), 3)
                      for c, p in zip(classes, y_proba)}

        return {
            "protocol":     label,
            "confidence":   round(probability, 3),
            "probabilities": proba_dict
        }

    # ─────────────────────────────────────────────
    # DISEASE MACRO ADJUSTMENT
    # ─────────────────────────────────────────────
    def apply_disease_rules(self, macros: dict, disease: str,
                             severity: str = "Mild") -> dict:
        """Adjust macros and return disease-specific diet rules."""
        if not disease or disease == "None":
            return {**macros, "disease_protocol": None, "disease_notes": [],
                    "max_sodium_mg": 2300, "disease_rules": {}}

        rules = DISEASE_RULES.get(disease)
        if not rules:
            return {**macros, "disease_protocol": None, "disease_notes": [],
                    "max_sodium_mg": 2300, "disease_rules": {}}

        sev_mult = SEVERITY_MULTIPLIERS.get(severity, 1.0)
        base_cal = macros["target_calories"]

        # Apply calorie adjustment scaled by severity
        adj_cal = max(1200, base_cal + rules["calorie_adj"] * sev_mult)
        p_pct   = rules["protein_pct"]
        c_pct   = rules["carb_pct"]
        f_pct   = rules["fat_pct"]

        adjusted = {
            "target_calories":   round(adj_cal),
            "target_protein_g":  round((adj_cal * p_pct) / 4),
            "target_carbs_g":    round((adj_cal * c_pct) / 4),
            "target_fat_g":      round((adj_cal * f_pct) / 9),
            "target_fiber_g":    35 if disease in ("Diabetes", "Obesity") else 28,
            "disease_protocol":  rules["protocol"],
            "disease_notes":     rules["notes"],
            "max_sodium_mg":     rules.get("max_sodium", 2300),
            "prefer_tags":       rules.get("prefer_tags", []),
            "avoid_tags":        rules.get("avoid_tags", []),
            "disease_rules":     rules,
        }
        return adjusted

    # ─────────────────────────────────────────────
    # TDEE  (Mifflin-St Jeor)
    # ─────────────────────────────────────────────
    def calculate_tdee(self, age, weight_kg, height_cm, gender, activity_level):
        if gender.lower() == "male":
            bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
        else:
            bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161

        multipliers = {
            "sedentary":        1.2,
            "lightly_active":   1.375,
            "moderately_active":1.55,
            "very_active":      1.725,
            "extra_active":     1.9,
        }
        return bmr * multipliers.get(activity_level, 1.2)

    # ─────────────────────────────────────────────
    # MACROS
    # ─────────────────────────────────────────────
    def calculate_macronutrients(self, tdee, goal="maintenance"):
        if goal == "lose_weight":
            target = tdee - 500
        elif goal == "build_muscle":
            target = tdee + 300
        elif goal == "aggressive_loss":
            target = tdee - 750
        else:
            target = tdee
        target = max(target, 1200)
        return {
            "target_calories":  round(target),
            "target_protein_g": round((target * 0.30) / 4),
            "target_carbs_g":   round((target * 0.40) / 4),
            "target_fat_g":     round((target * 0.30) / 9),
            "target_fiber_g":   28,
        }

    # ─────────────────────────────────────────────
    # BMI & BODY COMPOSITION
    # ─────────────────────────────────────────────
    def calculate_body_composition(self, age, weight_kg, height_cm, gender, bmi):
        if gender == "male":
            bf_pct = (1.20 * bmi) + (0.23 * age) - 16.2
        else:
            bf_pct = (1.20 * bmi) + (0.23 * age) - 5.4
        bf_pct    = max(5, min(60, round(bf_pct, 1)))
        fat_mass  = round(weight_kg * bf_pct / 100, 1)
        lean_mass = round(weight_kg - fat_mass, 1)

        if gender == "male":
            ideal_weight = 48 + 2.7 * ((height_cm - 152.4) / 2.54)
        else:
            ideal_weight = 45.5 + 2.2 * ((height_cm - 152.4) / 2.54)
        ideal_weight = round(max(40, ideal_weight), 1)

        return {
            "body_fat_pct":       bf_pct,
            "fat_mass_kg":        fat_mass,
            "lean_mass_kg":       lean_mass,
            "ideal_weight_kg":    ideal_weight,
            "weight_to_lose_gain": round(weight_kg - ideal_weight, 1),
        }

    # ─────────────────────────────────────────────
    # MICRONUTRIENT TARGETS
    # ─────────────────────────────────────────────
    def get_micronutrient_targets(self, age, gender, goal, disease=None):
        targets = {
            "Vitamin D (IU)":  600 if age < 70 else 800,
            "Calcium (mg)":    1000 if age < 50 else 1200,
            "Iron (mg)":       8 if gender == "male" or age > 50 else 18,
            "Omega-3 (g)":     1.6 if gender == "male" else 1.1,
            "Fiber (g)":       38 if gender == "male" else 25,
            "Potassium (mg)":  3400 if gender == "male" else 2600,
            "Vitamin C (mg)":  90 if gender == "male" else 75,
            "Magnesium (mg)":  420 if gender == "male" else 320,
        }
        if goal == "build_muscle":
            targets["Zinc (mg)"]  = 11 if gender == "male" else 8
            targets["B12 (mcg)"]  = 2.4
        if disease == "Diabetes":
            targets["Chromium (mcg)"] = 35
            targets["Magnesium (mg)"] = 450
        if disease in ("Hypertension", "Heart Disease"):
            targets["Potassium (mg)"]  = 4700
            targets["Omega-3 (g)"]     = 2.0 if gender == "male" else 1.6
            targets["Magnesium (mg)"]  = 500
        if disease == "Obesity":
            targets["Vitamin D (IU)"] = 800
        return targets

    # ─────────────────────────────────────────────
    # FOOD MENU – browse all available foods
    # ─────────────────────────────────────────────
    def get_food_menu(self, country=None, season=None, budget=None,
                       diet_type=None, disease=None, query=None,
                       page=1, page_size=50):
        """Return paginated food menu with health-suitability rating per food."""
        filtered = self._apply_filters(country, season, budget, diet_type)

        if query:
            filtered = filtered[
                filtered["name"].str.lower().str.contains(query.lower(), na=False)
            ]

        menu_items = []
        for _, row in filtered.iterrows():
            item = self._food_row_to_dict(row)
            item["suitability"] = self._rate_food_suitability(row, disease)
            menu_items.append(item)

        # Sort by suitability score desc, then name
        menu_items.sort(key=lambda x: (-x["suitability"]["score"], x["name"]))

        total = len(menu_items)
        start = (page - 1) * page_size
        end   = start + page_size

        return {
            "total":        total,
            "page":         page,
            "page_size":    page_size,
            "total_pages":  max(1, -(-total // page_size)),
            "items":        menu_items[start:end],
        }

    def _rate_food_suitability(self, row, disease=None):
        """Score a food 0–100 for general health + disease suitability."""
        score = 50.0
        flags = []

        # Nutrition density
        score += min(15, row.get("nutrient_density", 0) * 10)

        # Fiber
        if row.get("fiber", 0) >= 5:
            score += 10
            flags.append("High Fiber")

        # Sodium
        sodium = row.get("sodium_mg", 0)
        if sodium > 800:
            score -= 15
            flags.append("⚠️ High Sodium")
        elif sodium < 200:
            score += 5

        # GI
        gi = row.get("gi_score", 2)
        if gi == 1:
            score += 8
            flags.append("Low GI")
        elif gi == 3:
            score -= 8
            flags.append("High GI")

        # Disease-specific
        if disease:
            rules = DISEASE_RULES.get(disease, {})
            if row.get("diabetes_friendly_flag", 0) and disease == "Diabetes":
                score += 20
                flags.append("✅ Diabetes Safe")
            if row.get("bp_friendly_flag", 0) and disease == "Hypertension":
                score += 20
                flags.append("✅ BP Safe")
            if disease == "Hypertension" and sodium > 500:
                score -= 20
                flags.append("❌ High Sodium for BP")
            if disease == "Diabetes" and gi == 3:
                score -= 15
                flags.append("❌ High GI for Diabetes")
            if disease == "Heart Disease" and row.get("fat", 0) > 20:
                score -= 10
                flags.append("⚠️ High Fat")

        score = max(0, min(100, round(score, 1)))

        if score >= 80:
            rating = "Excellent"
        elif score >= 60:
            rating = "Good"
        elif score >= 40:
            rating = "Moderate"
        else:
            rating = "Avoid"

        return {"score": score, "rating": rating, "flags": flags}

    def _food_row_to_dict(self, row):
        return {
            "name":             str(row["name"]),
            "calories":         int(row.get("calories", 0)),
            "protein":          round(float(row.get("protein", 0)), 1),
            "carbs":            round(float(row.get("carbs", 0)), 1),
            "fat":              round(float(row.get("fat", 0)), 1),
            "fiber":            round(float(row.get("fiber", 0)), 1),
            "sodium_mg":        int(row.get("sodium_mg", 0)),
            "diet_type":        str(row.get("diet_type", "")),
            "country":          str(row.get("country", "")),
            "season":           str(row.get("season", "All")),
            "budget":           str(row.get("budget", "Low")),
            "micronutrients":   str(row.get("micronutrients", "")),
            "gi_level":         str(row.get("glycemic_index_val", "Medium")),
            "diabetes_friendly":bool(row.get("diabetes_friendly_flag", 0)),
            "bp_friendly":      bool(row.get("bp_friendly_flag", 0)),
            "meal_type":        str(row.get("meal_type_tag", "")),
            "health_goals":     str(row.get("health_goal_tag", "")),
            "allergens":        str(row.get("allergens_tag", "")),
            "health_safety_score": round(float(row.get("health_safety_score", 0)), 1),
            "nutrient_density": round(float(row.get("nutrient_density", 0)), 3),
        }

    # ─────────────────────────────────────────────
    # VALIDATE / RATE A USER-SELECTED FOOD LIST
    # ─────────────────────────────────────────────
    def validate_food_selection(self, food_names: list, disease=None,
                                  macros: dict = None):
        """
        Given a list of food names selected by the user, return per-food
        suitability ratings and aggregate nutrition vs. macro targets.
        """
        results = []
        totals  = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0, "fiber": 0, "sodium_mg": 0}

        for name in food_names:
            match = self.df[self.df["name"].str.lower() == name.lower()]
            if match.empty:
                results.append({
                    "name":   name,
                    "status": "not_found",
                    "suitability": {"score": 0, "rating": "Unknown", "flags": ["Not found in database"]},
                })
                continue

            row  = match.iloc[0]
            item = self._food_row_to_dict(row)
            item["suitability"] = self._rate_food_suitability(row, disease)
            item["status"]      = "found"

            for k in totals:
                totals[k] += float(row.get(k, 0))

            results.append(item)

        # Compare to macro targets
        comparison = {}
        if macros:
            comparison = {
                "calories_diff":  round(totals["calories"] - macros.get("target_calories", 0)),
                "protein_diff":   round(totals["protein"]  - macros.get("target_protein_g", 0), 1),
                "carbs_diff":     round(totals["carbs"]    - macros.get("target_carbs_g", 0), 1),
                "fat_diff":       round(totals["fat"]      - macros.get("target_fat_g", 0), 1),
                "sodium_mg_total":round(totals["sodium_mg"]),
                "within_calorie_target": abs(totals["calories"] - macros.get("target_calories", 0)) < 200,
            }

        return {
            "foods":       results,
            "totals":      {k: round(v, 1) for k, v in totals.items()},
            "comparison":  comparison,
        }

    # ─────────────────────────────────────────────
    # FILTERS
    # ─────────────────────────────────────────────
    def _apply_filters(self, country=None, season=None, budget=None, diet_type=None):
        filtered = self.df.copy()
        if country and country != "All":
            filtered = filtered[filtered["country"].isin([country, "All", "Global", "Pakistan"])]
        if season and season != "All":
            filtered = filtered[filtered["season"].isin([season, "All"])]
        if budget and budget != "All":
            order   = ["Low", "Medium", "High"]
            allowed = order[:order.index(budget) + 1] if budget in order else order
            filtered = filtered[filtered["budget"].isin(allowed)]
        if diet_type and diet_type != "All":
            types    = self.DIET_HIERARCHY.get(diet_type, list(self.DIET_HIERARCHY.keys()))
            filtered = filtered[filtered["diet_type"].isin(types)]

        # Fallback to full DB if too few
        if len(filtered) < 10:
            filtered = self.df.copy()
            if diet_type and diet_type != "All":
                types = self.DIET_HIERARCHY.get(diet_type, [])
                if types:
                    sub = self.df[self.df["diet_type"].isin(types)]
                    if len(sub) >= 5:
                        filtered = sub
        return filtered.reset_index(drop=True)

    def _apply_disease_filter(self, filtered: pd.DataFrame, disease: str) -> pd.DataFrame:
        """Boost safe foods, penalise unsafe ones (don't hard-filter – keep variety)."""
        if not disease:
            return filtered
        rules = DISEASE_RULES.get(disease, {})
        max_sodium = rules.get("max_sodium", 9999)
        filtered = filtered[filtered["sodium_mg"] <= max_sodium * 1.5].copy()
        if disease == "Diabetes":
            # Prefer low/medium GI
            filtered = filtered[filtered["gi_score"] <= 2].copy() if len(
                filtered[filtered["gi_score"] <= 2]) >= 5 else filtered
        return filtered.reset_index(drop=True)

    # ─────────────────────────────────────────────
    # RECOMMENDATION ENGINE
    # ─────────────────────────────────────────────
    def recommend_foods(self, target_calories, target_protein, target_carbs, target_fat,
                         country=None, season=None, budget=None, diet_type=None,
                         disease=None, n=5):
        filtered = self._apply_filters(country, season, budget, diet_type)
        filtered = self._apply_disease_filter(filtered, disease)
        if len(filtered) < 2:
            filtered = self.df.copy()

        tc = max(target_calories, 1)
        target_full = np.array([[
            target_calories, target_protein, target_fat, target_carbs,
            (target_protein * 4 / tc) * 100,
            (target_carbs   * 4 / tc) * 100,
            (target_fat     * 9 / tc) * 100,
            (target_protein * 4.0 - target_fat * 0.5 + target_carbs * 0.2) / tc,
            0, 0, 0, 2, 50  # fiber_density, sodium_density, gi_score, health_safety
        ]])

        X_f = self.scaler.transform(filtered[self.features])
        t_s = self.scaler.transform(target_full)

        k   = min(n + 15, len(filtered))
        knn = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="euclidean")
        knn.fit(X_f)
        distances, indices = knn.kneighbors(t_s)

        disease_rules = DISEASE_RULES.get(disease, {})
        prefer_flags  = disease_rules.get("prefer_tags", [])

        recs = []
        for dist, idx in zip(distances[0], indices[0]):
            food  = filtered.iloc[idx]
            score = (
                food["protein"]               * 4.0
                - abs(food["calories"] - target_calories) * 0.010
                - abs(food["protein"]  - target_protein)  * 0.5
                - food["fat"]                 * 0.2
                + food["carbs"]               * 0.15
                + food["nutrient_density"]    * 20.0
                + food["fiber"]               * 1.5
                + food["health_safety_score"] * 0.5
                - dist                        * 2.0
            )
            # Disease bonus
            if disease == "Diabetes" and food.get("diabetes_friendly_flag", 0):
                score += 15
            if disease == "Hypertension" and food.get("bp_friendly_flag", 0):
                score += 15
            if disease and food.get("sodium_mg", 0) < 300:
                score += 5

            suitability = self._rate_food_suitability(food, disease)

            recs.append({
                "name":              str(food["name"]),
                "calories":          int(food["calories"]),
                "protein":           round(float(food["protein"]), 1),
                "carbs":             round(float(food["carbs"]), 1),
                "fat":               round(float(food["fat"]), 1),
                "fiber":             round(float(food.get("fiber", 0)), 1),
                "sodium_mg":         int(food.get("sodium_mg", 0)),
                "diet_type":         str(food["diet_type"]),
                "budget":            str(food["budget"]),
                "score":             round(score, 2),
                "nutrient_density":  round(float(food["nutrient_density"]), 3),
                "protein_pct":       round(float(food["protein_pct"]), 1),
                "micronutrients":    str(food.get("micronutrients", "")),
                "gi_level":          str(food.get("glycemic_index_val", "Medium")),
                "diabetes_friendly": bool(food.get("diabetes_friendly_flag", 0)),
                "bp_friendly":       bool(food.get("bp_friendly_flag", 0)),
                "health_goals":      str(food.get("health_goal_tag", "")),
                "meal_type":         str(food.get("meal_type_tag", "")),
                "allergens":         str(food.get("allergens_tag", "")),
                "suitability":       suitability,
            })

        seen, unique = set(), []
        for r in sorted(recs, key=lambda x: x["score"], reverse=True):
            if r["name"] not in seen:
                seen.add(r["name"])
                unique.append(r)
            if len(unique) >= n:
                break
        return unique

    # ─────────────────────────────────────────────
    # SNACKS
    # ─────────────────────────────────────────────
    def recommend_snacks(self, macros, country=None, diet_type=None,
                          disease=None, n=4):
        snack_cals = macros["target_calories"] * 0.10
        cands = self.recommend_foods(
            snack_cals, snack_cals * 0.30 / 4,
            snack_cals * 0.40 / 4, snack_cals * 0.30 / 9,
            country=country, diet_type=diet_type, disease=disease, n=n + 10
        )
        snacks = [f for f in cands if f["calories"] <= 250][:n]
        if len(snacks) < n:
            snacks = cands[:n]
        return snacks

    # ─────────────────────────────────────────────
    # SINGLE-DAY MEAL PLAN
    # ─────────────────────────────────────────────
    def generate_meal_plan(self, macros, country=None, season=None, budget=None,
                            diet_type=None, disease=None, custom_foods=None):
        total_cal = macros["target_calories"]
        total_p   = macros["target_protein_g"]
        total_c   = macros["target_carbs_g"]
        total_f   = macros["target_fat_g"]
        config    = [
            ("breakfast", 0.25, "Breakfast"),
            ("lunch",     0.40, "Lunch"),
            ("dinner",    0.35, "Dinner"),
        ]
        preferred = set(f.lower() for f in (custom_foods or []))
        used, plan = set(), {}

        for key, ratio, label in config:
            cands = self.recommend_foods(
                total_cal * ratio, total_p * ratio,
                total_c * ratio, total_f * ratio,
                country=country, season=season, budget=budget,
                diet_type=diet_type, disease=disease, n=30
            )
            if preferred:
                priority = [f for f in cands if f["name"].lower() in preferred and f["name"] not in used]
                others   = [f for f in cands if f["name"].lower() not in preferred and f["name"] not in used]
                ordered  = priority + others
            else:
                ordered = [f for f in cands if f["name"] not in used]

            foods = ordered[:3]
            for f in foods:
                used.add(f["name"])

            plan[key] = {
                "label":           label,
                "target_calories": round(total_cal * ratio),
                "foods":           foods,
            }
        return plan

    # ─────────────────────────────────────────────
    # WEEKLY PLAN (7 days, no repeated meals)
    # ─────────────────────────────────────────────
    def generate_weekly_plan(self, macros, country=None, season=None, budget=None,
                              diet_type=None, disease=None, custom_foods=None):
        total_cal = macros["target_calories"]
        total_p   = macros["target_protein_g"]
        total_c   = macros["target_carbs_g"]
        total_f   = macros["target_fat_g"]
        meal_cfg  = [("breakfast", 0.25), ("lunch", 0.40), ("dinner", 0.35)]
        COOLDOWN  = 3

        slot_history = {mk: deque(maxlen=COOLDOWN * 3) for mk, _ in meal_cfg}
        weekly = []

        for day in range(1, 8):
            day_plan = {"day": day, "meals": {}}
            day_used = set()

            for meal_key, ratio in meal_cfg:
                candidates = self.recommend_foods(
                    total_cal * ratio, total_p * ratio,
                    total_c * ratio, total_f * ratio,
                    country=country, season=season, budget=budget,
                    diet_type=diet_type, disease=disease,
                    n=min(40, len(self.df))
                )
                slot_hist = slot_history[meal_key]
                foods = [f for f in candidates
                         if f["name"] not in slot_hist and f["name"] not in day_used][:3]

                if len(foods) < 3:
                    extra = [f for f in candidates
                             if f["name"] not in day_used and f["name"] not in {x["name"] for x in foods}]
                    foods = (foods + extra)[:3]

                if len(foods) < 3:
                    extra = [f for f in candidates if f["name"] not in {x["name"] for x in foods}]
                    foods = (foods + extra)[:3]

                for f in foods:
                    day_used.add(f["name"])
                    slot_hist.append(f["name"])

                day_plan["meals"][meal_key] = {
                    "label":           meal_key.capitalize(),
                    "target_calories": round(total_cal * ratio),
                    "foods":           foods,
                }

            day_plan["day_total_calories"] = sum(
                f["calories"] for m in day_plan["meals"].values() for f in m["foods"]
            )
            day_plan["day_total_protein"] = sum(
                f["protein"] for m in day_plan["meals"].values() for f in m["foods"]
            )
            weekly.append(day_plan)

        return weekly

    # ─────────────────────────────────────────────
    # PLAN FROM USER-SELECTED FOOD LIST
    # ─────────────────────────────────────────────
    def generate_meal_plan_from_list(self, macros, food_list, disease=None):
        food_names_lower = [f.lower() for f in food_list]
        subset = self.df[self.df["name"].str.lower().isin(food_names_lower)].copy()

        # Rate suitability
        if not subset.empty:
            subset["_suit"] = subset.apply(
                lambda r: self._rate_food_suitability(r, disease)["score"], axis=1
            )
            subset = subset.sort_values("_suit", ascending=False)

        if len(subset) == 0:
            return {"error": "None of the specified foods were found in the database."}

        total_cal = macros["target_calories"]
        config    = [
            ("breakfast", 0.25, "Breakfast"),
            ("lunch",     0.40, "Lunch"),
            ("dinner",    0.35, "Dinner"),
        ]
        used, plan = set(), {}

        for key, ratio, label in config:
            target     = total_cal * ratio
            candidates = subset[~subset["name"].isin(used)].copy()
            candidates["_diff"] = (candidates["calories"] - target).abs()
            top   = candidates.nsmallest(3, "_diff")
            foods = []
            for _, row in top.iterrows():
                used.add(row["name"])
                item = self._food_row_to_dict(row)
                item["suitability"] = self._rate_food_suitability(row, disease)
                foods.append(item)
            plan[key] = {"label": label, "target_calories": round(target), "foods": foods}

        return plan

    # ─────────────────────────────────────────────
    # DURATION ROADMAP
    # ─────────────────────────────────────────────
    def recommend_duration(self, goal, weight_kg, height_cm, bmi, disease=None):
        if goal == "lose_weight":
            if bmi >= 30:
                weeks = 24 if disease else 20
                expected = f"-{round(weeks * 0.5 * 0.8, 1)} kg"
                phases = [
                    {"name": "Foundation",    "weeks": "1–4",   "focus": "Eliminate processed foods, establish routine, 500 kcal deficit", "icon": "🌱"},
                    {"name": "Active Loss",   "weeks": "5–16",  "focus": "Consistent deficit + 3–4× cardio/week, weekly weigh-ins",         "icon": "🔥"},
                    {"name": "Consolidation", "weeks": "17–"+str(weeks), "focus": "Reduce to 300 kcal deficit, introduce maintenance eating",  "icon": "⚖️"},
                ]
                tip = "Obese range: target 0.5–0.75 kg/week. Sustainable loss preserves muscle."
            elif bmi >= 25:
                weeks = 14 if disease else 12
                expected = f"-{round(weeks * 0.45 * 0.8, 1)} kg"
                phases = [
                    {"name": "Kick-start", "weeks": "1–3",  "focus": "Cut sugar & refined carbs, walk 30 min daily", "icon": "🚀"},
                    {"name": "Loss",       "weeks": f"4–{weeks-2}", "focus": "500 kcal deficit + 3× cardio/week, track every meal", "icon": "📉"},
                    {"name": "Taper",      "weeks": f"{weeks-1}–{weeks}", "focus": "Reduce to 200 kcal deficit, begin maintenance habits", "icon": "🎯"},
                ]
                tip = "Overweight range: 12–14 week plan targets ~4–6 kg fat loss safely."
            else:
                weeks = 8
                expected = f"-{round(weeks * 0.3 * 0.8, 1)} kg"
                phases = [
                    {"name": "Lean-out",         "weeks": "1–5", "focus": "Mild 300 kcal deficit, keep protein ≥1.6g/kg", "icon": "💪"},
                    {"name": "Maintenance Prep",  "weeks": "6–8", "focus": "Return to TDEE, track body composition weekly",  "icon": "🏆"},
                ]
                tip = "Normal-weight cut: protein is your top priority."
            check_in = "Weigh in every Monday morning, fasted. Track trends not daily fluctuations."

        elif goal == "build_muscle":
            weeks    = 16
            expected = f"+{round(weeks * 0.15, 1)} kg lean mass"
            phases   = [
                {"name": "Hypertrophy I",  "weeks": "1–6",   "focus": "Progressive overload 3–4×/week, 300 kcal surplus", "icon": "💪"},
                {"name": "Hypertrophy II", "weeks": "7–12",  "focus": "Increase training volume, track strength PRs",        "icon": "🏋️"},
                {"name": "Consolidation",  "weeks": "13–16", "focus": "Deload in week 14, maintain surplus",                 "icon": "📊"},
            ]
            tip      = "Muscle growth: expect ~0.5–1 kg lean mass/month with consistent training."
            check_in = "Weigh weekly. Take progress photos every 4 weeks."

        else:  # maintenance
            weeks    = 8
            expected = "±0 kg"
            phases   = [
                {"name": "Calibration", "weeks": "1–3", "focus": "Match intake to TDEE precisely", "icon": "🎯"},
                {"name": "Habit",       "weeks": "4–8", "focus": "Maintain weight within ±1 kg",    "icon": "✅"},
            ]
            tip      = "Reassess every 8 weeks and adjust for activity changes."
            check_in = "Weekly weigh-in to catch early drift."

        disease_note = ""
        if disease and disease != "None":
            disease_note = (f"⚕️ {disease} management: follow protocol consistently throughout "
                            f"all phases and review with your physician every 4 weeks.")

        return {
            "recommended_weeks": weeks,
            "expected_change":   expected,
            "phases":            phases,
            "tip":               tip,
            "check_in":          check_in,
            "disease_note":      disease_note,
            "hydration_ml":      round(weight_kg * 35),
            "review_at_weeks":   sorted(set([
                max(1, round(weeks * 0.25)),
                max(2, round(weeks * 0.5)),
                max(3, round(weeks * 0.75)),
                weeks,
            ])),
        }

    # ─────────────────────────────────────────────
    # HEALTH INSIGHTS
    # ─────────────────────────────────────────────
    def generate_health_insights(self, age, weight_kg, height_cm, gender,
                                  bmi, tdee, macros, goal, activity,
                                  disease=None, glucose=None,
                                  cholesterol=None, systolic_bp=None):
        insights  = []
        warnings_ = []

        # BMI
        if bmi < 18.5:
            warnings_.append("⚠️ Underweight – avoid deficit. Focus on nutrient-dense, calorie-rich foods.")
        elif bmi >= 30:
            insights.append("🎯 Sustainable loss (0.5 kg/week) beats crash dieting – you'll retain more muscle.")
        elif bmi >= 25:
            insights.append("📉 A consistent 500 kcal daily deficit produces ~0.5 kg/week fat loss.")

        # Protein
        prot_per_kg = macros["target_protein_g"] / weight_kg
        if prot_per_kg < 1.2:
            insights.append(
                f"💪 Protein ({macros['target_protein_g']}g/day) is below optimal. "
                f"Try increasing to {round(weight_kg * 1.6)}g to preserve lean mass."
            )
        else:
            insights.append(f"✅ Protein target ({macros['target_protein_g']}g = {prot_per_kg:.1f}g/kg) is well-calibrated.")

        # Age
        if age >= 50:
            insights.append("🦴 Over 50: prioritise calcium, Vitamin D, and resistance training for bone density.")
        elif age < 20:
            insights.append("🌱 Under 20: do not go below 1500 kcal/day – adolescent growth needs adequate energy.")

        # Activity
        if activity == "sedentary":
            insights.append("🚶 Adding a 20-min walk daily raises TDEE by ~100 kcal and improves insulin sensitivity.")
        elif activity == "very_active":
            insights.append("⚡ High activity: ensure carbohydrates are sufficient for performance recovery.")

        # Sleep
        if age < 18:
            insights.append("😴 Teens need 8–10 hours of sleep for optimal hormone regulation.")
        elif age < 65:
            insights.append("😴 Aim for 7–9 hours of sleep – poor sleep increases hunger hormones by up to 24%.")
        else:
            insights.append("😴 Adults 65+ benefit from 7–8 hours sleep for metabolic health.")

        # Disease-specific blood indicators
        if disease == "Diabetes":
            if glucose and glucose > 126:
                warnings_.append(f"🩸 Fasting glucose {glucose} mg/dL is in diabetic range. Strictly limit high-GI foods.")
            elif glucose and glucose > 100:
                warnings_.append(f"🩸 Fasting glucose {glucose} mg/dL is pre-diabetic. Prioritise low-GI foods.")
            insights.append("🥗 Eat every 3–4 hrs in small portions to keep blood sugar stable.")
            insights.append("🚫 Avoid: white rice, white bread, sugary drinks, sweets, fried foods.")

        if disease == "Hypertension":
            if systolic_bp and systolic_bp > 140:
                warnings_.append(f"💓 BP {systolic_bp}+ mmHg is Stage 2 hypertension. Medical supervision required.")
            elif systolic_bp and systolic_bp > 130:
                warnings_.append(f"💓 BP {systolic_bp} mmHg – limit sodium strictly (<1500 mg/day).")
            insights.append("🧂 Use herbs and spices instead of salt for flavouring.")
            insights.append("🫐 DASH superfoods: berries, spinach, oats, low-fat dairy, potatoes.")

        if disease == "Heart Disease":
            if cholesterol and cholesterol > 240:
                warnings_.append(f"💊 Total cholesterol {cholesterol} mg/dL is high. Limit saturated fats, increase soluble fiber.")
            insights.append("🐟 Eat fatty fish 2× per week (salmon, sardines) for cardioprotective Omega-3.")
            insights.append("🫒 Replace butter/ghee with olive oil or avocado oil.")

        if disease == "Obesity":
            insights.append("🍽️ Use a smaller plate (20–25 cm) – studies show it reduces portions by 22%.")
            insights.append("💧 Drink 500 ml of water 30 minutes before meals to reduce calorie intake.")

        hydration = round(weight_kg * 35)
        insights.append(f"💧 Water target: {hydration} ml/day ({hydration/1000:.1f}L). Add 500 ml/hr of exercise.")
        insights.append("🌾 Target 25–38g fiber daily – improves satiety, gut health, and blood sugar control.")

        return {
            "insights":     insights,
            "warnings":     warnings_,
            "hydration_ml": hydration,
        }

    # ─────────────────────────────────────────────
    # HYDRATION PLAN
    # ─────────────────────────────────────────────
    def get_hydration_plan(self, weight_kg, activity, disease=None, temperature="Moderate"):
        base_ml    = round(weight_kg * 35)
        extra_ml   = 500 if activity in ("very_active", "extra_active") else 250
        if temperature == "Hot":
            extra_ml += 500

        total_ml = base_ml + extra_ml

        schedule = [
            {"time": "On waking",        "ml": 300, "note": "Kickstart metabolism"},
            {"time": "Before breakfast",  "ml": 200, "note": "Aid digestion"},
            {"time": "Mid-morning",       "ml": 300, "note": "Sustain energy"},
            {"time": "Before lunch",      "ml": 200, "note": "Portion control"},
            {"time": "Afternoon",         "ml": 300, "note": "Prevent energy dip"},
            {"time": "Before dinner",     "ml": 200, "note": "Reduce meal intake"},
            {"time": "Evening",           "ml": 200, "note": "Avoid sleeping thirsty"},
        ]

        if disease == "Hypertension":
            schedule.append({"time": "Any time", "ml": 0,
                              "note": "Limit sodium in beverages – avoid sports drinks unless exercising"})
        if disease == "Diabetes":
            schedule.append({"time": "Any time", "ml": 0,
                              "note": "Avoid sugar-sweetened beverages entirely"})

        return {
            "base_ml":        base_ml,
            "activity_extra": extra_ml,
            "total_ml":       total_ml,
            "total_litres":   round(total_ml / 1000, 1),
            "schedule":       schedule,
            "electrolytes":   disease in ("Hypertension",) or activity == "very_active",
        }

    # ─────────────────────────────────────────────
    # KNN SIMILAR FOODS
    # ─────────────────────────────────────────────
    def find_similar_foods(self, food_name: str, n=5, disease=None):
        """Find nutritionally similar foods using KNN."""
        match = self.df[self.df["name"].str.lower() == food_name.lower()]
        if match.empty:
            return {"error": f"'{food_name}' not found in database."}

        idx    = match.index[0]
        x_food = self.X_all[idx:idx+1]
        _, indices = self.knn_model.kneighbors(x_food, n_neighbors=n + 5)

        similar = []
        for j in indices[0]:
            if j == idx:
                continue
            row  = self.df.iloc[j]
            item = self._food_row_to_dict(row)
            item["suitability"] = self._rate_food_suitability(row, disease)
            similar.append(item)
            if len(similar) >= n:
                break

        return {"query": food_name, "similar_foods": similar}


# ═══════════════════════════════════════════════════════════
#  QUICK SMOKE TEST
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os, sys
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    engine = DietRecommendationEngine()

    # --- Basic TDEE + Macros ---
    tdee   = engine.calculate_tdee(35, 85, 170, "male", "moderately_active")
    macros = engine.calculate_macronutrients(tdee, "lose_weight")

    # --- Disease adjustment ---
    adj = engine.apply_disease_rules(macros, "Diabetes", severity="Moderate")
    print(f"\n📊 TDEE: {round(tdee)} | Target: {macros['target_calories']} → Diabetic adj: {adj['target_calories']} kcal")
    print(f"📋 Protocol: {adj['disease_protocol']}")

    # --- Health-risk prediction ---
    risk = engine.predict_health_risk(35, 85, 170, "male", "Moderate",
                                       glucose=140, cholesterol=220,
                                       systolic_bp=145, severity="Moderate")
    print(f"\n🧠 Risk prediction: {risk['protocol']} (confidence {risk['confidence']:.1%})")

    # --- Food menu ---
    menu = engine.get_food_menu(country="Pakistan", disease="Diabetes", page_size=5)
    print(f"\n🍽️ Menu sample (Diabetes-safe, Pakistan):")
    for item in menu["items"][:3]:
        print(f"  {item['name']:30s} | {item['calories']} kcal | {item['suitability']['rating']}")

    # --- Meal plan ---
    plan = engine.generate_meal_plan(adj, country="Pakistan", diet_type="Non-Veg", disease="Diabetes")
    print("\n📅 Day Meal Plan:")
    for meal, details in plan.items():
        print(f"  {details['label']}: {', '.join(f['name'] for f in details['foods'])}")

    print("\n✅ All systems operational.")