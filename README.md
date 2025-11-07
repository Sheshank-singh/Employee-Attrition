# Employee Attrition Prediction System

![Employee Attrition](https://img.shields.io/badge/Project-ML-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-lightgrey)
![College Project](https://img.shields.io/badge/Type-College_Project-orange)

## Project Overview

The Employee Attrition Prediction System is a machine learning-based web application that helps HR professionals predict employee attrition risk. This project was developed as a college group project to demonstrate the practical application of machine learning in solving real-world business problems.

### Key Features

- Interactive web interface for data input
- Real-time attrition prediction
- Detailed analysis of contributing factors
- HR-focused recommendations
- Visual representation of predictions
- Mobile-responsive design

## Technology Stack

### Frontend
- HTML5
- CSS3 with modern design principles
- Font Awesome for icons
- Responsive web design

### Backend
- Python
- Flask web framework
- SHAP (SHapley Additive exPlanations) for model interpretability
- Scikit-learn for machine learning

### Machine Learning Implementation

#### Model Development
We implemented and evaluated three supervised learning models after applying SMOTE balancing:

1. **Logistic Regression (Selected Model)**
   - Chosen for interpretability and strong generalization
   - Used scaled features for optimal performance
   - Best overall performance across metrics

2. **Random Forest Classifier**
   - Implemented for ensemble-based predictions
   - Provided feature importance insights
   - Strong accuracy but lower recall

3. **Decision Tree Classifier**
   - Baseline model with rule-based interpretability
   - Simpler model for comparison

#### Training Methodology
- Data Split: 80% training / 20% testing using stratified sampling
- Feature Engineering: 18 carefully selected employee attributes
  - Personal factors (Age, Marital Status)
  - Professional factors (Job Role, Years at Company)
  - Satisfaction metrics
  - Work-life balance indicators
- Preprocessing:
  - SMOTE for class imbalance
  - Feature scaling (for Logistic Regression)
  - Standardized data preparation

## 📊 Dataset

The project uses the HR Employee Attrition dataset, which includes:
- 1,470 employee records
- 18 relevant features
- Balanced class distribution
- Comprehensive employee metrics

## 🚀 Getting Started

### Prerequisites
```bash
python 3.8 or higher
pip (Python package manager)
```

### Installation Steps

1. Clone the repository
```bash
git clone https://github.com/Sheshank-singh/Employee-Attrition.git
cd Employee-Attrition
```

2. Install required packages
```bash
pip install -r Backend/requirements.txt
```

3. Run the application
```bash
cd Backend
python app.py
```

4. Access the application
```
Open http://localhost:5000 in your web browser
```

## 💡 Usage Guide

1. Navigate to the home page
2. Fill in the employee information form
3. Submit the form to get predictions
4. Review the detailed analysis and recommendations

## 📌 Project Structure

```
Employee-Attrition/
├── Backend/
│   ├── model/
│   ├── app.py
│   ├── requirements.txt
│   └── HR-Employee-Attrition.csv
├── Frontend/
│   ├── static/
│   │   ├── style.css
│   │   ├── result.css
│   │   └── landing.css
│   └── templates/
│       ├── index.html
│       ├── result.html
│       └── dashboard.html
└── README.md
```

## 🌟 Features in Detail

### 1. Prediction Capabilities
- Real-time attrition risk assessment using Logistic Regression
- Probability scores with 83.6% accuracy
- Key factor identification using SHAP values
- Balanced predictions through SMOTE implementation

### 2. User Interface
- Clean, modern design
- Intuitive form layout
- Mobile-responsive interface
- Section-wise organized input fields

### 3. Analysis Output
- Visual representation of results
- Detailed factor breakdown
- HR-specific recommendations
- Easy-to-understand metrics

## 📈 Model Performance & Selection

### Comparative Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|---------|----------|----------|
| Logistic Regression | 0.836 | 0.486 | 0.382 | 0.428 | 0.653 |
| Random Forest | 0.816 | 0.387 | 0.255 | 0.307 | 0.589 |
| Decision Tree | 0.734 | 0.245 | 0.319 | 0.277 | 0.566 |

### Model Selection Rationale
Logistic Regression was chosen as the production model due to:
- Highest accuracy at 83.6%
- Superior recall and F1 scores for attrition prediction
- Better ROC-AUC performance (0.653)
- Excellent interpretability for HR professionals
- Well-generalized predictions for new data

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
*Note: This project was developed as an academic exercise and should be used accordingly.*
