<h1 align="center">💳 Fraud Detection + AI Analyst (Streamlit App)</h1>
<p align="center">
An intelligent, AI-powered fraud detection and analysis system built using Streamlit, LightGBM, and LLMs
</p>

---

### 🧑‍🏫 Guided By  
<p align="center">
<b>Prof. Jeffrey Shaffer</b> — Director, Applied AI Lab & First Reader, University of Cincinnati  
<br>
<b>Prof. Lucas Timothy</b> — Assistant Professor of Business Analytics Dept. & Second Reader, University of Cincinnati  
</p>

---

## 🚀 Project Overview

The **Fraud Detection + AI Analyst** app is an **interactive fraud risk analysis tool** that combines **machine learning**, **cost-sensitive optimization**, and **AI-based interpretation** to help financial analysts detect fraudulent transactions more effectively.

### 🎯 Core Objectives
- ⚡ **Predict fraudulent transactions** using LightGBM & Logistic Regression  
- 📉 **Optimize decision thresholds** via cost-based evaluation (False Positives vs False Negatives)  
- 🧩 **Interpret metrics automatically** through an integrated AI Analyst (LLM)  
- 📊 **Visualize performance** using ROC, Precision-Recall, Confusion Matrix, and Feature Importance plots  
- 📄 **Generate PDF summaries** of AI-driven recommendations  

Built for **financial institutions and risk teams**, the app transforms raw transaction data into actionable insights — combining automation, explainability, and visual storytelling.

---

## 🧰 Tech Stack

| Technology | Description |
|------------|-------------|
| ![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white) | Core scripting and ML pipeline |
| ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white) | Interactive dashboard framework |
| ![LightGBM](https://img.shields.io/badge/LightGBM-9DDE8B?logo=lightning&logoColor=black) | Gradient boosting model for fraud prediction |
| ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?logo=scikit-learn&logoColor=white) | Model training, cross-validation, metrics |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white) | Data cleaning and transformation |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white) | Numerical processing |
| ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white) | Interactive data visualization |
| ![ReportLab](https://img.shields.io/badge/ReportLab-lightgrey?logo=adobeacrobatreader&logoColor=red) | PDF report generation |
| ![OpenAI](https://img.shields.io/badge/OpenAI%2FGroq-412991?logo=openai&logoColor=white) | LLM-powered AI Analyst module |

---

## 🔁 Workflow Overview

### 1️⃣ Upload Data
- Accepts either:
  - A merged CSV file with **`isFraud`** column, or  
  - Separate **Transaction.csv** and **Identity.csv** files  
- Automatically merges, validates, and prepares input

### 2️⃣ Preprocess Data
- Cleans column names and removes unnecessary fields  
- Encodes categorical variables using **OrdinalEncoder**  
- Standardizes binary target column (`isFraud`)

### 3️⃣ Train Model
- Trains using **LightGBM** or **Logistic Regression**  
- Supports **Time-Aware K-Fold** and **Stratified K-Fold** cross-validation  
- Generates **Out-of-Fold (OOF)** predictions for honest AUC evaluation

### 4️⃣ Evaluate Performance
- Visualizes:
  - 🧮 Confusion Matrix  
  - 📈 ROC Curve with AUC  
  - 📊 Precision-Recall Curve  
- Displays precision, recall, accuracy, FPR, and cost metrics

### 5️⃣ Threshold Optimization
- Lets users define **False Positive** and **False Negative** costs  
- Automatically identifies the **cost-optimal threshold**  
- Provides downloadable CSV of all threshold statistics

### 6️⃣ Feature Importance
- Displays **top predictive features** using interactive bar plots  
- Helps analysts understand which transaction attributes drive fraud risk

### 7️⃣ AI Analyst
- Integrates with **OpenAI / Groq LLM APIs**  
- Auto-interprets model metrics & recommends next actions  
- Users can ask natural-language questions like:  
  > “What threshold minimizes cost given our FP/FN ratio?”  
  > “Which features contribute most to false positives?”  
- Generates a **PDF report** summarizing key AI recommendations

---

## 📈 Key Outputs
| Output | Description |
|--------|--------------|
| 🧮 **Confusion Matrix** | Interactive heatmap with true vs predicted outcomes |
| 📊 **ROC Curve** | Model discrimination with AUC score |
| 📉 **Precision-Recall Curve** | Trade-off visualization between recall and precision |
| 🧱 **Feature Importance** | Ranked impact of key predictors |
| 🤖 **AI Analyst Report (PDF)** | Actionable AI summary and recommendations |

---

## 🖥️ How to Run

### 🔧 Install Dependencies

```bash

pip install streamlit scikit-learn lightgbm pandas numpy plotly requests pydantic reportlab
```
## ▶️ Launch the App

```bash

streamlit run "Fraud Detection + AI Analyst — Streamlit (Sujith-custom).py"
```

🔑 (Optional) Set API Credentials

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="https://api.groq.com/openai"
export OPENAI_MODEL="groq/compound"
```
