# Employee Turnover Prediction Model
**Model Version**: `GradientBoostingClassifier v3.3 (Weighted Duplicates)` 
**Author**: _Dzmitry Kandrykinski_ 
**Date**: _2025-07-07_

---

## Overview  
This project provides a machine learning model to predict employee turnover (binary classification: `left` = 1, `stayed` = 0). 
The model is trained using historical HR data and optimized for high recall in identifying employees at risk of leaving.

---

## Files Included

| File Name                                        | Description                                |
|--------------------------------------------------|--------------------------------------------|
| `final_HR_turnover_model_pipeline_template.pkl`  | Trained Gradient Boosting model wrapped in a preprocessing pipeline |
| `sample_input_structure.csv`                     | Sample input data in the required format   |
| `README.md`                                      | This guide                                 |
| `preprocessing.py`                               | Preprocessing code used in pipeline        |
| `requirements.txt`                               | Dependencies                               |

---

## Model Summary

- **Model Type**: `GradientBoostingClassifier`
- **Technique**: Duplicate row grouping + sample weighting
- **Cross-validation**: `StratifiedKFold (5-fold)`
- **Preprocessing**:
  - Renamed `sales` ➝ `department`
  - Scaled `average_monthly_hours` (divided by 100)
  - One-hot encoded `department` and `salary`
- **Target Variable**: `left` (0 = stayed, 1 = left)

---

## Model Performance

| Metric        | Value |
|---------------|-------|
| Accuracy      | 0.98  |
| Precision (1) | 0.96  |
| Recall (1)    | 0.93  |
| F1-score (1)  | 0.94  |
| Confusion Matrix | `[[11337, 91], [238, 3333]]` |

---

## Input Data Format

| Column Name             | Type    | Description                               |
|-------------------------|---------|-------------------------------------------|
| satisfaction_level      | float   | From 0 to 1                               |
| last_evaluation         | float   | From 0 to 1                               |
| number_project          | int     | Number of projects                        |
| average_monthly_hours   | int     | Will be scaled automatically (÷ 100)     |
| time_spend_company      | int     | Years at company                          |
| Work_accident           | int     | 0 = No, 1 = Yes                           |
| promotion_last_5years   | int     | 0 = No, 1 = Yes                           |
| sales                   | string  | e.g., "sales", "support", "technical"     |
| salary                  | string  | "low", "medium", "high"                   |

**Note**: Do not include the target variable column (may cause errors if included).

**Note**: Do not pre-encode or scale. The pipeline handles this automatically.

**Note**: May be useful to run two-sample T-test to be sure the duplicate rows represent
systematically different groups rather than random duplicates. 

---
### Hypothesis:

* H0 (Null Hypothesis): The duplicated rows are random and do not represent a meaningful statistical pattern.
* H1 (Alternative Hypothesis): The duplicated rows are systematically DIFFERENT — representing a group of 
people with shared features (e.g., same job, performance level, etc.).

```python
from scipy.stats import ttest_ind, chi2_contingency

# Step 1: Identify Duplicates Non-Duplicates
duplicates = df[df.duplicated(keep=False)]
non_duplicates = df[~df.duplicated(keep=False)]
print(f"Duplicates: {len(duplicates)}, Non-Duplicates: {len(non_duplicates)}")

# Step 2: Split and Test
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
cat_cols = df.select_dtypes(include='object').columns

# T-tests for numerical features
print("\n----- T-test results for numerical features: -----\n")
for col in num_cols:
    _, pval = ttest_ind(duplicates[col], non_duplicates[col], equal_var=False)
    print(f"{col}: p-value = {pval:.8f} {'< 0.05 => DIFFERENT (Rejrct the H0 hypothesis)' if pval < 0.05 else '=> similar (no significant difference)'}")


# Duplicate rows flag - 1
df['duplicate_flag'] = df.duplicated(keep=False).astype(int)
df['duplicate_flag'].value_counts()

# Chi-square test for categorical features. One-way chi-square on full column
print("\n----- Chi-square test results for categorical features: -----\n")

for col in cat_cols:
    contingency = pd.crosstab(df['duplicate_flag'], df[col])
    if contingency.shape[0] > 1 and contingency.shape[1] > 1:
        chi2, pval, _, _ = chi2_contingency(contingency)
        print(f"{col}: p-value = {pval:.4f} {'< 0.05 => DIFFERENT (Rejrct the H0 hypothesis)' if pval < 0.05 else '=> similar (no significant difference)'}")
    else:
        print(f"{col}: skipped (not enough variation for test)")
```

If the test confirms Ho-hypothesis: Duplicate rows are random and do not represent a 
significant statistical pattern - drop the duplicate rows before using the model.

If the test rejects Ho-hypothesis, it means that the duplicate rows represent a group of people
with shared features (e.g., same job, performance level, etc.) - leave them in the data set.

---

## Model Usage Example

```python
import pandas as pd
import joblib
from preprocessing import basic_preprocess


# Load model
pipeline = joblib.load('final_HR_turnover_model_pipeline_template.pkl')

# Load new data
new_data = pd.read_csv('new_employees.csv')  # Must include raw columns listed above

# Predict probabilities
proba = pipeline.predict_proba(new_data)[:, 1]

# Predict binary classification (0/1)
preds = pipeline.predict(new_data)

# Add results to DataFrame
new_data['turnover_probability'] = proba
new_data['predicted_left'] = preds
```

