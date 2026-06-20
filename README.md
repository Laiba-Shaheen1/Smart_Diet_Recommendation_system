# 🥗 NutriAI – Smart Diet Recommendation System

A **machine learning-powered personalized diet planning application** with disease-aware nutritional guidance, meal planning, and AI chatbot support. NutriAI combines modern web technology with intelligent nutrition science to deliver customized diet plans tailored to individual health profiles.

---

## ✨ Features

### 🎯 Core Functionality
- **Personalized Diet Planning** – Generate custom nutrition plans based on age, weight, height, gender, activity level, and fitness goals
- **Disease-Aware Recommendations** – Special protocols for Diabetes, Hypertension, Obesity, and Heart Disease
- **TDEE & Macro Calculation** – Precise calorie and macronutrient targets (protein, carbs, fat)
- **BMI & Body Composition Analysis** – Calculate body fat %, lean mass, and ideal weight targets
- **ML Health Risk Prediction** – AI-powered assessment based on glucose, cholesterol, and blood pressure
- **Micronutrient Targeting** – Age and gender-specific vitamin & mineral recommendations

### 🍽️ Meal Planning
- **Daily & Weekly Meal Plans** – AI-generated, disease-appropriate meal suggestions
- **Food Menu Browsing** – Paginated, filterable food database with disease suitability ratings
- **Custom Food Selection** – Build meal plans from user-selected foods
- **Nutritional Validation** – Check selected foods against health conditions and macro targets
- **Similar Food Swaps** – Find nutritionally equivalent alternatives (KNN-based)

### 💧 Additional Guidance
- **Hydration Plans** – Personalized water intake recommendations based on activity and disease
- **Programme Duration** – Timeline for achieving fitness goals with phase-based progression
- **Health Insights** – AI-generated warnings and actionable nutrition advice
- **Food Database Management** – Add, remove, and search foods with metadata (country, season, budget, diet type)

### 🤖 AI Assistant
- **Claude AI Integration** – Real-time nutrition Q&A and meal plan explanations
- **Contextual Chat** – Chatbot understands user's health profile and diet restrictions
- **Suggestion Prompts** – Quick-start templates for common nutrition questions

---

## 📋 Supported Conditions
| Condition | Protocol |
|-----------|----------|
| **Diabetes** | Low GI, controlled carbs, fiber-rich |
| **Hypertension** | Low sodium, potassium-rich, DASH diet principles |
| **Obesity** | Caloric deficit, high protein, satiety-focused |
| **Heart Disease** | Low saturated fat, omega-3 rich, fiber emphasis |

---

## 🛠️ Tech Stack

### **Backend**
- **Python 3.x**
- **Flask** – REST API framework
- **Pandas** – Data manipulation & CSV handling
- **Scikit-learn** – ML-based health risk prediction
- **Anthropic Claude API** – AI chatbot integration

### **Frontend**
- **HTML5** – Semantic markup
- **CSS3** – Modern styling with CSS variables
- **JavaScript (ES6+)** – Interactive UI & API integration
- **Chart.js** – Macronutrient visualization

### **Data**
- CSV-based food database with nutritional metadata
- Calorie calculator dataset
- Patient health dataset for model training

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip (Python package manager)
