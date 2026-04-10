# Predicting Under-Five Mortality for Vaccine Distribution Optimisation

**Course:** Real Life Machine Learning — Tilburg University (2025)  
**Team:** Salwa & Sude (Group 16)  
**Tools:** Python, scikit-learn, SHAP, pandas, seaborn, matplotlib

---

## The Problem

Vaccine resources are scarce globally, and international health organisations often have to decide years in advance where to allocate them. Better predictions of child mortality risk could help direct vaccines to where they are needed most.

This project asks: **can we predict under-five mortality rates from country-level health and socioeconomic indicators — and if so, which factors matter most?**

---

## What We Did

### Data
We used a real-world dataset of country-level health indicators including vaccination coverage (Measles, Polio, Diphtheria, Hepatitis B), GDP, schooling, healthcare expenditure, and infant deaths. The target variable was **under-five deaths per country per year**.

### Preprocessing
- Identified and handled missing values through imputation
- Removed multicollinear features (e.g. infant deaths, which overlap with the target)
- Combined Hepatitis B coverage across genders into a single average feature
- Scaled all features before modelling
- Applied Spearman correlation analysis to understand feature relationships

### Models Compared
| Model | RMSE | R² |
|---|---|---|
| Polynomial Regression (degree 2) | 157.94 | 0.29 |
| Ridge Regression (RidgeCV) | — | -0.34 |
| Lasso Regression (LassoCV) | — | -0.02 |
| **Random Forest (tuned)** | **lowest** | **best** |

Random Forest significantly outperformed all linear models. Ridge and Lasso performed worse than simply predicting the mean, confirming that the relationships in this dataset are non-linear.

### Hyperparameter Tuning
Random Forest was tuned using `RandomizedSearchCV` with 5-fold cross-validation, optimising number of trees, max depth, and features per split.

### Feature Importance (SHAP)
SHAP analysis revealed that **Measles vaccination coverage** was by far the strongest predictor (importance: 0.43), followed by Polio coverage and schooling. Countries with low vaccination rates were consistently predicted to have higher child mortality.

---

## Key Findings

- Linear models failed to capture the complexity of child mortality drivers
- Random Forest generalised well for low-mortality countries but underestimated mortality for high-risk regions — a known limitation with skewed targets
- Vaccination coverage and education level were the most influential predictors, consistent with public health literature

---

## Honest Reflection

The dataset's overall correlations with the target were weaker than expected, suggesting unmeasured factors (conflict, infrastructure, access to care) also play a major role. The model is a useful starting point for allocation planning but would benefit from richer data and potentially a more complex ensemble approach.

---

## My Contribution

This was a two-person group project. I contributed to the exploratory data analysis, feature selection decisions, model evaluation, and interpretation of SHAP results.

---

## Files
- `ML_Project.ipynb` — full notebook with code, analysis, and visualisations
- 
