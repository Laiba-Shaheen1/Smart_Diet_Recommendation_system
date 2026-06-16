"""
NutriAI – Flask REST API
=========================
All existing endpoints preserved + new disease-aware, food-menu,
food-validation, and similar-food endpoints.
"""

from flask import Flask, request, jsonify
try:
    from flask_cors import CORS
    HAS_CORS = True
except ImportError:
    HAS_CORS = False

from model import DietRecommendationEngine
import pandas as pd
import os

app = Flask(__name__)
if HAS_CORS:
    CORS(app)

DATA_PATH         = "foods_data.csv"
CALORIE_DATA_PATH = "calorie_calculator_diet_dataset.csv"
PATIENT_DATA_PATH = "diet_recommendations_dataset_.csv"

engine = DietRecommendationEngine(
    data_path=DATA_PATH,
    calorie_data_path=CALORIE_DATA_PATH,
    patient_data_path=PATIENT_DATA_PATH,
)


def _reload_engine():
    global engine
    engine = DietRecommendationEngine(
        data_path=DATA_PATH,
        calorie_data_path=CALORIE_DATA_PATH,
        patient_data_path=PATIENT_DATA_PATH,
    )


# ═══════════════════════════════════════════════════════════
#  CORE DIET ENDPOINT  (enhanced with disease support)
# ═══════════════════════════════════════════════════════════
@app.route("/get_diet", methods=["POST"])
def get_diet():
    """
    Full diet analysis with optional disease / blood-indicator parameters.

    Body (JSON):
        age, weight, height, gender, activity, goal          – required
        country, season, budget, diet_type, custom_foods     – optional
        disease        : "Diabetes" | "Hypertension" | "Obesity" | "Heart Disease" | null
        severity       : "Mild" | "Moderate" | "Severe"
        glucose        : fasting blood glucose (mg/dL)
        cholesterol    : total cholesterol (mg/dL)
        systolic_bp    : systolic blood pressure (mmHg)
        exercise_hours : hours of exercise/week
    """
    data = request.json
    try:
        age      = int(data["age"])
        weight   = float(data["weight"])
        height   = float(data["height"])
        gender   = data["gender"].lower()
        activity = data["activity"]
        goal     = data["goal"]

        country   = data.get("country",   "Pakistan")
        season    = data.get("season",    "All")
        budget    = data.get("budget",    "Low")
        diet_type = data.get("diet_type", "Non-Veg")

        # Disease / health indicators
        disease     = data.get("disease",     None)
        if not disease or disease in ("", "None"):
            disease = None
        severity    = data.get("severity",    "Mild")
        glucose     = data.get("glucose",     None)
        cholesterol = data.get("cholesterol", None)
        systolic_bp = data.get("systolic_bp", None)
        ex_hours    = float(data.get("exercise_hours", 3))
        nu_imbalance= float(data.get("nutrient_imbalance", 2))

        custom_foods = data.get("custom_foods", [])

        # ── Core calculations ──
        tdee   = engine.calculate_tdee(age, weight, height, gender, activity)
        macros = engine.calculate_macronutrients(tdee, goal)

        bmi = round(weight / (height / 100) ** 2, 1)
        bmi_category = (
            "Underweight" if bmi < 18.5 else
            "Normal"      if bmi < 25   else
            "Overweight"  if bmi < 30   else
            "Obese"
        )

        body_comp      = engine.calculate_body_composition(age, weight, height, gender, bmi)
        micronutrients = engine.get_micronutrient_targets(age, gender, goal, disease)
        duration       = engine.recommend_duration(goal, weight, height, bmi, disease)
        hydration      = engine.get_hydration_plan(weight, activity, disease)

        # ── Disease adjustment ──
        adjusted_macros = engine.apply_disease_rules(macros, disease, severity) if disease else macros

        # ── Health-risk ML prediction ──
        risk = engine.predict_health_risk(
            age, weight, height, gender, activity,
            glucose        = glucose     or 100,
            cholesterol    = cholesterol or 180,
            systolic_bp    = systolic_bp or 120,
            caloric_intake = adjusted_macros["target_calories"],
            exercise_hours = ex_hours,
            nutrient_imbalance = nu_imbalance,
            severity       = severity,
        )

        # ── Health insights ──
        health = engine.generate_health_insights(
            age, weight, height, gender, bmi, tdee,
            adjusted_macros, goal, activity,
            disease=disease, glucose=glucose,
            cholesterol=cholesterol, systolic_bp=systolic_bp,
        )

        # ── Meal plans ──
        meal_plan   = engine.generate_meal_plan(
            adjusted_macros, country, season, budget, diet_type,
            disease=disease, custom_foods=custom_foods,
        )
        weekly_plan = engine.generate_weekly_plan(
            adjusted_macros, country, season, budget, diet_type,
            disease=disease, custom_foods=custom_foods,
        )

        # ── Top food recommendations ──
        foods = engine.recommend_foods(
            adjusted_macros["target_calories"] * 0.35,
            adjusted_macros["target_protein_g"] * 0.35,
            adjusted_macros["target_carbs_g"]   * 0.35,
            adjusted_macros["target_fat_g"]     * 0.35,
            country, season, budget, diet_type,
            disease=disease, n=8,
        )
        snacks = engine.recommend_snacks(
            adjusted_macros, country, diet_type, disease=disease, n=4
        )

        return jsonify({
            "status":           "success",
            "tdee":             round(tdee),
            "bmi":              bmi,
            "bmi_category":     bmi_category,
            "macros":           macros,
            "adjusted_macros":  adjusted_macros,
            "body_comp":        body_comp,
            "micronutrients":   micronutrients,
            "disease":          disease,
            "severity":         severity,
            "health_risk":      risk,
            "meal_plan":        meal_plan,
            "weekly_plan":      weekly_plan,
            "foods":            foods,
            "snacks":           snacks,
            "duration":         duration,
            "hydration":        hydration,
            "health":           health,
            "metrics":          engine.metrics,
        })

    except Exception as e:
        import traceback
        return jsonify({"status": "error", "message": str(e),
                        "trace": traceback.format_exc()}), 400


# ═══════════════════════════════════════════════════════════
#  FOOD MENU  (paginated, filterable, with suitability rating)
# ═══════════════════════════════════════════════════════════
@app.route("/food_menu", methods=["GET"])
def food_menu():
    """
    Browse paginated food menu with health-suitability scores.

    Query params:
        country, season, budget, diet_type  – filters
        disease                             – rate each food for this disease
        q                                   – name search query
        page, page_size
    """
    try:
        country   = request.args.get("country")
        season    = request.args.get("season")
        budget    = request.args.get("budget")
        diet_type = request.args.get("diet_type")
        disease   = request.args.get("disease")
        query     = request.args.get("q", "")
        page      = int(request.args.get("page", 1))
        page_size = int(request.args.get("page_size", 50))

        result = engine.get_food_menu(
            country=country, season=season, budget=budget,
            diet_type=diet_type, disease=disease,
            query=query, page=page, page_size=page_size,
        )
        return jsonify({"status": "success", **result})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# ═══════════════════════════════════════════════════════════
#  VALIDATE FOOD SELECTION
# ═══════════════════════════════════════════════════════════
@app.route("/validate_foods", methods=["POST"])
def validate_foods():
    """
    User submits a list of chosen foods → get per-food suitability + macro comparison.

    Body (JSON):
        food_names : list of food names
        disease    : optional
        macros     : optional target macros dict
    """
    data = request.json
    try:
        food_names = data.get("food_names", [])
        disease    = data.get("disease")
        macros     = data.get("macros")

        result = engine.validate_food_selection(food_names, disease=disease, macros=macros)
        return jsonify({"status": "success", **result})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# ═══════════════════════════════════════════════════════════
#  SIMILAR FOODS (KNN)
# ═══════════════════════════════════════════════════════════
@app.route("/similar_foods", methods=["GET"])
def similar_foods():
    """Find nutritionally similar foods to a given food."""
    try:
        food_name = request.args.get("name", "")
        n         = int(request.args.get("n", 5))
        disease   = request.args.get("disease")

        result = engine.find_similar_foods(food_name, n=n, disease=disease)
        return jsonify({"status": "success", **result})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# ═══════════════════════════════════════════════════════════
#  HEALTH-RISK PREDICTION  (standalone)
# ═══════════════════════════════════════════════════════════
@app.route("/predict_risk", methods=["POST"])
def predict_risk():
    """
    ML-based diet protocol prediction from health indicators.

    Body: age, weight, height, gender, activity, severity,
          glucose, cholesterol, systolic_bp, exercise_hours, nutrient_imbalance
    """
    data = request.json
    try:
        result = engine.predict_health_risk(
            age             = int(data["age"]),
            weight_kg       = float(data["weight"]),
            height_cm       = float(data["height"]),
            gender          = data["gender"],
            activity        = data["activity"],
            glucose         = float(data.get("glucose",         100)),
            cholesterol     = float(data.get("cholesterol",     180)),
            systolic_bp     = float(data.get("systolic_bp",     120)),
            caloric_intake  = float(data.get("caloric_intake",  2000)),
            exercise_hours  = float(data.get("exercise_hours",  3)),
            nutrient_imbalance = float(data.get("nutrient_imbalance", 2)),
            severity        = data.get("severity", "Mild"),
        )
        return jsonify({"status": "success", "prediction": result})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# ═══════════════════════════════════════════════════════════
#  HYDRATION PLAN
# ═══════════════════════════════════════════════════════════
@app.route("/hydration", methods=["POST"])
def hydration():
    data = request.json
    try:
        plan = engine.get_hydration_plan(
            weight_kg   = float(data["weight"]),
            activity    = data["activity"],
            disease     = data.get("disease"),
            temperature = data.get("temperature", "Moderate"),
        )
        return jsonify({"status": "success", "hydration": plan})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# ═══════════════════════════════════════════════════════════
#  PLAN FROM SELECTED FOODS
# ═══════════════════════════════════════════════════════════
@app.route("/plan_from_foods", methods=["POST"])
def plan_from_foods():
    data = request.json
    try:
        age       = int(data["age"])
        weight    = float(data["weight"])
        height    = float(data["height"])
        gender    = data["gender"].lower()
        activity  = data["activity"]
        goal      = data["goal"]
        food_list = data.get("food_list", [])
        disease   = data.get("disease")

        if not food_list:
            return jsonify({"status": "error",
                            "message": "Provide at least one food name in 'food_list'."}), 400

        tdee   = engine.calculate_tdee(age, weight, height, gender, activity)
        macros = engine.calculate_macronutrients(tdee, goal)
        if disease:
            macros = engine.apply_disease_rules(macros, disease, data.get("severity", "Mild"))
        bmi = round(weight / (height / 100) ** 2, 1)

        plan = engine.generate_meal_plan_from_list(macros, food_list, disease=disease)

        return jsonify({
            "status":   "success",
            "tdee":     round(tdee),
            "bmi":      bmi,
            "macros":   macros,
            "disease":  disease,
            "meal_plan": plan,
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# ═══════════════════════════════════════════════════════════
#  FOOD CRUD  (preserved from original)
# ═══════════════════════════════════════════════════════════
@app.route("/get_foods", methods=["GET"])
def get_foods():
    try:
        df    = pd.read_csv(DATA_PATH)
        foods = df.to_dict(orient="records")
        return jsonify({"status": "success", "foods": foods, "total": len(foods)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/add_food", methods=["POST"])
def add_food():
    data = request.json
    try:
        required = ["name", "calories", "protein", "carbs", "fat"]
        for field in required:
            if field not in data:
                return jsonify({"status": "error",
                                "message": f"Missing required field: {field}"}), 400

        df = pd.read_csv(DATA_PATH)
        if data["name"].strip().lower() in df["name"].str.lower().values:
            return jsonify({"status": "error",
                            "message": f"Food '{data['name']}' already exists."}), 409

        new_row = {
            "name":           data["name"].strip(),
            "calories":       float(data["calories"]),
            "protein":        float(data["protein"]),
            "carbs":          float(data["carbs"]),
            "fat":            float(data["fat"]),
            "diet_type":      data.get("diet_type",  "Non-Veg"),
            "country":        data.get("country",    "Global"),
            "season":         data.get("season",     "All"),
            "budget":         data.get("budget",     "Low"),
            "fiber":          float(data.get("fiber",     0)),
            "sodium_mg":      float(data.get("sodium_mg", 0)),
            "micronutrients": data.get("micronutrients", ""),
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(DATA_PATH, index=False)
        _reload_engine()

        return jsonify({
            "status":       "success",
            "message":      f"'{new_row['name']}' added successfully.",
            "total_foods":  len(df),
            "food":         new_row,
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/remove_food", methods=["DELETE"])
def remove_food():
    data = request.json
    try:
        food_name = data.get("name", "").strip()
        if not food_name:
            return jsonify({"status": "error", "message": "Food name required."}), 400

        df   = pd.read_csv(DATA_PATH)
        mask = df["name"].str.lower() == food_name.lower()
        if not mask.any():
            return jsonify({"status": "error",
                            "message": f"'{food_name}' not found."}), 404

        df = df[~mask]
        df.to_csv(DATA_PATH, index=False)
        _reload_engine()

        return jsonify({
            "status":      "success",
            "message":     f"'{food_name}' removed.",
            "total_foods": len(df),
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/search_foods", methods=["GET"])
def search_foods():
    try:
        query     = request.args.get("q",         "").lower()
        diet_type = request.args.get("diet_type", "")
        country   = request.args.get("country",   "")
        budget    = request.args.get("budget",    "")
        disease   = request.args.get("disease",   "")

        df = engine.df.copy()

        if query:
            df = df[df["name"].str.lower().str.contains(query, na=False)]
        if diet_type:
            df = df[df["diet_type"].str.lower() == diet_type.lower()]
        if country:
            df = df[df["country"].str.lower().isin([country.lower(), "all", "global"])]
        if budget:
            order   = ["low", "medium", "high"]
            allowed = order[:order.index(budget.lower()) + 1] if budget.lower() in order else order
            df = df[df["budget"].str.lower().isin(allowed)]

        items = []
        for _, row in df.head(50).iterrows():
            item = engine._food_row_to_dict(row)
            if disease:
                item["suitability"] = engine._rate_food_suitability(row, disease)
            items.append(item)

        return jsonify({"status": "success", "foods": items, "total": len(df)})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# ═══════════════════════════════════════════════════════════
#  HOME
# ═══════════════════════════════════════════════════════════
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status":  "running",
        "message": "NutriAI Enhanced API is active",
        "version": "2.0.0",
        "endpoints": {
            "core": [
                "POST /get_diet",
                "POST /predict_risk",
                "POST /hydration",
            ],
            "foods": [
                "GET  /food_menu?country=&season=&budget=&diet_type=&disease=&q=&page=&page_size=",
                "POST /validate_foods",
                "GET  /similar_foods?name=&n=&disease=",
                "GET  /search_foods?q=&diet_type=&country=&budget=&disease=",
                "GET  /get_foods",
                "POST /add_food",
                "DELETE /remove_food",
            ],
            "planning": [
                "POST /plan_from_foods",
            ],
        },
        "supported_diseases": ["Diabetes", "Hypertension", "Obesity", "Heart Disease"],
        "model_metrics":      engine.metrics,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5003)
