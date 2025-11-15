# Customer Churn Prediction

A machine learning project to predict customer churn for a telecommunications company using Python, scikit-learn, and data visualization tools.

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

## 📊 Project Overview

**Goal:** Build a predictive model to identify customers likely to churn and surface key drivers for retention strategies.

**Dataset:** Telco Customer Churn dataset with 7,043 customer records and 21 features including demographics, account information, and service subscriptions.

**Key Results:**
- 🎯 **Model Performance:** ROC-AUC of 0.84 achieved with both Logistic Regression and Random Forest
- 📈 **Churn Rate:** 26.54% of customers churning
- 🔍 **Top Predictors:** Tenure, Total Charges, Monthly Charges, Contract Type

## 🚀 Quick Start

### Prerequisites
- Python 3.12 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Launch Jupyter notebook:
```bash
jupyter notebook churnanalysis_cleaned.ipynb
```

## 📁 Project Structure

```
customer-churn-prediction/
├── churnanalysis_cleaned.ipynb    # Main analysis notebook
├── Customer-Churn.csv             # Dataset (not included in repo)
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── .gitignore                     # Git ignore rules
```

## 🔍 Analysis Workflow

1. **Data Loading & Inspection**
   - Load 7,043 customer records
   - Examine data structure and types

2. **Data Cleaning & Preprocessing**
   - Convert TotalCharges to numeric
   - Handle missing values
   - Encode categorical variables

3. **Exploratory Data Analysis (EDA)**
   - Visualize churn distribution
   - Analyze numeric features (tenure, charges)
   - Examine categorical features (contract type, services)

4. **Feature Engineering**
   - Binary encoding for Yes/No features
   - One-hot encoding for multi-category features
   - Feature scaling with StandardScaler

5. **Model Training**
   - Logistic Regression (baseline)
   - Random Forest Classifier
   - Train-test split (80/20)

6. **Model Evaluation**
   - Classification metrics (precision, recall, F1)
   - ROC-AUC curves
   - Confusion matrices
   - Feature importance analysis

## 📈 Key Findings

### Model Performance
| Model | ROC-AUC | Accuracy | Precision (Churn) | Recall (Churn) |
|-------|---------|----------|-------------------|----------------|
| Logistic Regression | 0.842 | 81% | 66% | 57% |
| Random Forest | 0.840 | 80% | 67% | 53% |

### Top Churn Drivers
1. **Tenure** (17.9% importance) - Shorter tenure = higher churn
2. **Total Charges** (17.6% importance) - Lower total spend correlates with churn
3. **Monthly Charges** (13.2% importance) - Higher monthly fees increase churn
4. **Contract Type** - Month-to-month contracts have significantly higher churn
5. **Internet Service** - Fiber optic customers show elevated churn rates
6. **Payment Method** - Electronic check users churn more frequently

## 💡 Business Recommendations

1. **Contract Incentives**: Offer discounts for customers to upgrade from month-to-month to 1-year or 2-year contracts
2. **Early Intervention**: Implement onboarding programs for customers in their first 12 months
3. **Pricing Strategy**: Review pricing for high monthly charge segments; consider loyalty discounts
4. **Payment Experience**: Encourage automatic payment methods; improve electronic check experience
5. **Service Quality**: Monitor and enhance fiber optic service satisfaction

## 🛠️ Technologies Used

- **Python 3.14**
- **Data Analysis:** pandas, numpy
- **Machine Learning:** scikit-learn
- **Visualization:** matplotlib, seaborn
- **Environment:** Jupyter Notebook

## 📊 Dataset

The dataset includes the following features:
- Customer demographics (gender, senior citizen, partner, dependents)
- Account information (tenure, contract type, payment method, billing)
- Services (phone, internet, security, backup, support, streaming)
- Target variable: Churn (Yes/No)

**Note:** The dataset (`Customer-Churn.csv`) is not included in this repository. You can obtain similar datasets from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

## 🔮 Future Enhancements

- [ ] Deploy model as REST API using Flask/FastAPI
- [ ] Create interactive dashboard with Streamlit or Dash
- [ ] Implement SMOTE for handling class imbalance
- [ ] Experiment with XGBoost and other ensemble methods
- [ ] Add hyperparameter tuning with GridSearchCV
- [ ] Create automated model retraining pipeline

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

Your Name
- GitHub: [@Data-Analytics-repo](https://github.com/Data-Analytics-repo/Customer-churn.git)
- LinkedIn: [Azimil kabir Shaikh](www.linkedin.com/in/azimil-shaikh-854b0328b)

## 🙏 Acknowledgments

- Dataset source: Telco Customer Churn dataset
- Inspiration from various Kaggle notebooks and data science communities
