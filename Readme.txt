Project Title: Fraud Detection + AI Analyst — Streamlit (Sujith-custom)
Author: Sarath Sai Sujith Srinivas Grandhe
University: University of Cincinnati, MS in Business Analytics
Date: October 2025

------------------------------------------------------------
ABOUT THE PROJECT
------------------------------------------------------------
This project presents an advanced fraud detection application built using Streamlit and Python.
The app implements machine learning models (Logistic Regression and LightGBM) to predict the probability
that an online transaction is fraudulent. It features a complete end-to-end ML workflow — from preprocessing 
and model evaluation to interpretability using interactive charts and AI-assisted insights.

The "AI Analyst" module automatically interprets evaluation metrics, confusion matrices, and ROC/PR curves 
to provide human-readable explanations for the model’s behavior. This feature adds transparency and helps 
non-technical stakeholders understand the results.

------------------------------------------------------------
KEY FEATURES
------------------------------------------------------------
1. Data Preprocessing:
   - Missing value handling, categorical encoding, and feature engineering.
   - Leak-safe target encoding and time-aware cross-validation support.

2. Model Training:
   - Implements Logistic Regression and LightGBM with K-Fold and TimeBlocked CV.
   - Out-of-Fold predictions for honest evaluation and better generalization.

3. Evaluation:
   - ROC and Precision-Recall curves with dynamic thresholding.
   - Confusion Matrix and detailed metric tables (Accuracy, Precision, Recall, F1, AUC).

4. AI Analyst:
   - Automatically generates insights about the model’s strengths and weaknesses.
   - Summarizes findings for decision-makers using contextual analysis.

5. Streamlit Dashboard:
   - Tabs for data upload, preprocessing, model results, and AI interpretation.
   - Clean interface with visual diagnostics and downloadable outputs.

------------------------------------------------------------
TECH STACK
------------------------------------------------------------
Languages: Python
Libraries: Streamlit, Scikit-learn, LightGBM, Matplotlib, NumPy, Pandas
Version Control: GitHub
Environment: Python 3.10+

------------------------------------------------------------
HOW TO RUN THE APP
------------------------------------------------------------
1. Install dependencies

2. Run the Streamlit app:
   streamlit run "Fraud Detection + AI Analyst — Streamlit (Sujith-custom).py"

3. Upload the dataset(s) when prompted in the app interface.

4. The app will process the data, train the model, and visualize key metrics automatically.

------------------------------------------------------------
FILE STRUCTURE
------------------------------------------------------------
|-- app.py                               # Main Streamlit app
|-- data/                                # (Optional) Sample dataset folder
|-- outputs/                             # Folder for saved model results
|-- README.txt                           # Project description (this file)

------------------------------------------------------------
ACKNOWLEDGMENTS
------------------------------------------------------------
This project was developed under the guidance of:
- Prof. Jeffrey Shaffer (First Reader)
- Prof. Lucas Timothy (Second Reader)

Special thanks to the University of Cincinnati Business Analytics faculty for their continuous support and knowledge.

------------------------------------------------------------
CONTACT
------------------------------------------------------------
Author: Sarath Sai Sujith Srinivas Grandhe  
Email: sarathsai.grandhe@gmail.com  
LinkedIn: www.linkedin.com/in/sujithgrandhe  
GitHub: https://github.com/Sarathsai4
