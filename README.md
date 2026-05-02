Proiect Machine Learning — Sisteme Inteligente

Acest proiect a fost realizat în cadrul cursului de Sisteme Inteligente și acoperă întregul pipeline de Machine Learning: analiză exploratorie (EDA), preprocesare, antrenare modele, optimizare hiperparametri, evaluare și explicabilitate.

Sunt abordate două tipuri de probleme:

Clasificare: predicția bolii cardiace (Heart Disease UCI Dataset)
Regresie: predicția prețurilor locuințelor (King County House Sales)

Instalare și rulare
Clonează repository-ul
Instalează dependențele:
pip install -r requirements.txt
Rulează notebook-urile pentru a genera modelele:
notebooks/regresie.ipynb
notebooks/clasificare.ipynb
Pornește aplicația:
cd app
streamlit run app.py
Tehnologii utilizate
Python (pandas, numpy, matplotlib, seaborn)
Scikit-learn
XGBoost, CatBoost, EBM
SHAP
Streamlit
Joblib
Modele comparate

Clasificare:

Naive Bayes
Logistic Regression
Decision Tree
Random Forest
SVM
K-Nearest Neighbors
XGBoost
CatBoost
EBM

Regresie:

Linear Regression
Decision Tree
Random Forest
SVR
KNN Regressor
Gaussian Process
XGBoost
CatBoost
EBM
Rezultate
Clasificare: Random Forest, XGBoost și CatBoost obțin performanțe foarte ridicate (ROC-AUC până la 1.0)
Regresie: CatBoost oferă cel mai bun scor R² (~0.893)
Notă

Folderul models/ nu este inclus în repository. Modelele trebuie generate local prin rularea notebook-urilor.
