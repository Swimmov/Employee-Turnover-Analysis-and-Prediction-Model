# 🧠 Employee Turnover Prediction Model
**Model Version**: `GradientBoostingClassifier v3.3 (Weighted Duplicates)`  
**Author**: _Dzmitry Kandrykinski_  
**Date**: _2025-07-07_

---

## 📌 Overview  
This project provides a machine learning model to predict employee turnover (binary classification: `left` = 1, `stayed` = 0).  
The model is trained using historical HR data and optimized for high recall in identifying employees at risk of leaving.

---

## 📂 Files Included

| File Name                                        | Description                                |
|--------------------------------------------------|--------------------------------------------|
| `final_HR_turnover_model_pipeline_template.pkl`  | Trained Gradient Boosting model wrapped in a preprocessing pipeline |
| `sample_input_structure.csv`                     | Sample input data in the required format   |
| `README.md`                                      | This guide                                 |
| `preprocessing.py`                               | Preprocessing code used in pipeline        |
| `requirements.txt`                               | dependencies                               |

---

## 🧠 Model Summary

- **Model Type**: `GradientBoostingClassifier`
- **Technique**: Duplicate row grouping + sample weighting
- **Cross-validation**: `StratifiedKFold (5-fold)`
- **Preprocessing**:
  - Renamed `sales` ➝ `department`
  - Scaled `average_monthly_hours` (divided by 100)
  - One-hot encoded `department` and `salary`
- **Target Variable**: `left` (0 = stayed, 1 = left)

---

## 📊 Model Performance

| Metric        | Value |
|---------------|-------|
| Accuracy      | 0.98  |
| Precision (1) | 0.96  |
| Recall (1)    | 0.93  |
| F1-score (1)  | 0.94  |
| Confusion Matrix | `[[11337, 91], [238, 3333]]` |

---

## 💾 Input Data Format

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

⚠️ **Note**: Do not include the target variable column (may cause errors if included).
⚠️ **Note**: Do not pre-encode or scale. The pipeline handles this automatically.
---

## 🧪 Usage Example

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

## 📬 Contact & Support

If you have any questions or want to integrate this into an application, feel free to contact:  
📧 your.email@example.com  
💼 [LinkedIn Profile or GitHub]
