from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import os
import plotly.express as px
import shap

# Flask setup
app = Flask(__name__, template_folder="../frontend/templates", static_folder="../frontend/static")

# ============================
# Model Prediction Code
# ============================

# Define paths
model_path = os.path.join(os.path.dirname(__file__), "model", "logistic_regression_attrition.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "model", "scaler.pkl")

# Load model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Feature columns expected by the model
feature_columns = [
    'Age', 'DistanceFromHome', 'EnvironmentSatisfaction', 'JobInvolvement',
    'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'StockOptionLevel',
    'TotalWorkingYears', 'WorkLifeBalance', 'YearsAtCompany',
    'YearsInCurrentRole', 'YearsWithCurrManager', 'OverTime_Yes',
    'JobRole_Human Resources', 'JobRole_Laboratory Technician',
    'JobRole_Manager', 'JobRole_Manufacturing Director',
    'JobRole_Research Director', 'JobRole_Research Scientist',
    'JobRole_Sales Executive', 'JobRole_Sales Representative',
    'MaritalStatus_Married', 'MaritalStatus_Single',
    'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
    'EducationField_Life Sciences', 'EducationField_Marketing',
    'EducationField_Medical', 'EducationField_Other',
    'EducationField_Technical Degree'
]


# Initialize SHAP Explainer
explainer = shap.Explainer(model, scaler.transform(pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)))

def predict_attrition(
    Age, DistanceFromHome, EnvironmentSatisfaction, JobInvolvement, JobLevel,
    JobSatisfaction, MonthlyIncome, StockOptionLevel, TotalWorkingYears,
    WorkLifeBalance, YearsAtCompany, YearsInCurrentRole,
    YearsWithCurrManager, OverTime, JobRole, MaritalStatus,
    BusinessTravel, EducationField
):
    input_data = {col: 0 for col in feature_columns}

    # Fill numeric values
    input_data.update({
        'Age': Age,
        'DistanceFromHome': DistanceFromHome,
        'EnvironmentSatisfaction': EnvironmentSatisfaction,
        'JobInvolvement': JobInvolvement,
        'JobLevel': JobLevel,
        'JobSatisfaction': JobSatisfaction,
        'MonthlyIncome': MonthlyIncome,
        'StockOptionLevel': StockOptionLevel,
        'TotalWorkingYears': TotalWorkingYears,
        'WorkLifeBalance': WorkLifeBalance,
        'YearsAtCompany': YearsAtCompany,
        'YearsInCurrentRole': YearsInCurrentRole,
        'YearsWithCurrManager': YearsWithCurrManager,
    })

    # One-hot encodings
    if OverTime == "Yes":
        input_data['OverTime_Yes'] = 1

    jobrole_col = f"JobRole_{JobRole}"
    if jobrole_col in input_data:
        input_data[jobrole_col] = 1

    if MaritalStatus != "Divorced":
        marital_col = f"MaritalStatus_{MaritalStatus}"
        input_data[marital_col] = 1

    travel_col = f"BusinessTravel_{BusinessTravel}"
    if travel_col in input_data:
        input_data[travel_col] = 1

    edu_col = f"EducationField_{EducationField}"
    if edu_col in input_data:
        input_data[edu_col] = 1

    # Scale and predict
    df_input = pd.DataFrame([input_data])
    df_input_scaled = pd.DataFrame(scaler.transform(df_input), columns=df_input.columns)
    df_input_raw = df_input.copy()  # Keep unscaled values for human-readable output
    
    # Make prediction
    prediction = model.predict(df_input_scaled)[0]
    probability = model.predict_proba(df_input_scaled)[0][1]
    
    # Get SHAP values for explanation
    shap_values = explainer.shap_values(df_input_scaled)[0]

    result = "Yes (Attrition Likely)" if prediction == 1 else "No (Attrition Unlikely)"
    reasons, suggestions = generate_explanation_and_suggestions(df_input_raw, shap_values, prediction)
    return result, probability, reasons, suggestions

    result = "Yes (Attrition Likely)" if prediction == 1 else "No (Attrition Unlikely)"
    reasons, suggestions = generate_explanation_and_suggestions(df_input, shap_values, prediction)
    return result, probability, reasons, suggestions


def generate_explanation_and_suggestions(input_data, shap_values, prediction):
    feature_names = input_data.columns
    
    # Get top 3 features contributing to the prediction
    # For attrition (prediction == 1), we look for positive SHAP values
    # For no attrition (prediction == 0), we look for negative SHAP values
    if prediction == 1:
        # Sort by positive SHAP values for attrition
        top_features_indices = np.argsort(shap_values)[::-1][:3]
    else:
        # Sort by negative SHAP values for no attrition
        top_features_indices = np.argsort(shap_values)[:3]

    reasons = []
    suggestions = []

    for i in top_features_indices:
        feature = feature_names[i]
        shap_value = shap_values[i]
        feature_value = input_data[feature].iloc[0]

        reason_text = ""
        suggestion_text = ""

        # Map features to human-readable reasons and HR suggestions
        if feature == 'MonthlyIncome':
            if shap_value > 0: # Contributes to attrition
                reason_text = f"Low Monthly Income: The employee's monthly income of ${feature_value:,.0f} is a significant factor."
                suggestion_text = "Review salary structure and consider competitive compensation adjustments."
            else: # Contributes to no attrition
                reason_text = f"High Monthly Income: The employee's monthly income of ${feature_value:,.0f} is a positive factor."
                suggestion_text = "Maintain competitive compensation to retain talent."
        elif feature == 'OverTime_Yes':
            if feature_value == 1 and shap_value > 0:
                reason_text = "Overtime Work: The employee frequently works overtime."
                suggestion_text = "Encourage better work-life balance, review workload, and consider flexible schedules."
            elif feature_value == 0 and shap_value < 0:
                reason_text = "No Overtime Work: The employee does not work overtime."
                suggestion_text = "Promote healthy work-life balance initiatives."
        elif feature == 'JobSatisfaction':
            if shap_value > 0:
                reason_text = f"Low Job Satisfaction: The employee's job satisfaction is rated {int(feature_value)} (out of 4)."
                suggestion_text = "Introduce employee engagement programs, conduct satisfaction surveys, and address concerns."
            else:
                reason_text = f"High Job Satisfaction: The employee's job satisfaction is rated {int(feature_value)} (out of 4)."
                suggestion_text = "Continue fostering a positive work environment and recognize contributions."
        elif feature == 'EnvironmentSatisfaction':
            if shap_value > 0:
                reason_text = f"Low Environment Satisfaction: The employee's environment satisfaction is rated {int(feature_value)} (out of 4)."
                suggestion_text = "Improve workplace conditions, address environmental concerns, and gather feedback."
            else:
                reason_text = f"High Environment Satisfaction: The employee's environment satisfaction is rated {int(feature_value)} (out of 4)."
                suggestion_text = "Maintain a supportive and comfortable work environment."
        elif feature == 'YearsAtCompany':
            if shap_value > 0:
                reason_text = f"Fewer Years at Company: The employee has only been with the company for {int(feature_value)} years."
                suggestion_text = "Implement mentorship programs and career development plans for newer employees."
            else:
                reason_text = f"More Years at Company: The employee has been with the company for {int(feature_value)} years."
                suggestion_text = "Recognize long-term contributions and offer growth opportunities."
        elif feature == 'Age':
            if shap_value > 0:
                reason_text = f"Younger Age: The employee's age ({int(feature_value)} years) might be a factor."
                suggestion_text = "Provide career growth opportunities and development programs for younger talent."
            else:
                reason_text = f"Older Age: The employee's age ({int(feature_value)} years) is a stabilizing factor."
                suggestion_text = "Value experience and offer opportunities for senior employees to mentor."
        elif feature == 'DistanceFromHome':
            if shap_value > 0:
                reason_text = f"Long Commute: The employee lives {int(feature_value)} miles from work."
                suggestion_text = "Consider remote work options or relocation assistance if feasible."
            else:
                reason_text = f"Short Commute: The employee lives {int(feature_value)} miles from work."
                suggestion_text = "A convenient commute contributes to employee satisfaction."
        elif feature == 'JobLevel':
            if shap_value > 0:
                reason_text = f"Lower Job Level: The employee is at Job Level {int(feature_value)}."
                suggestion_text = "Offer clear career progression paths and promotion opportunities."
            else:
                reason_text = f"Higher Job Level: The employee is at Job Level {int(feature_value)}."
                suggestion_text = "Ensure challenging work and opportunities for leadership."
        elif feature == 'TotalWorkingYears':
            if shap_value > 0:
                reason_text = f"Fewer Total Working Years: The employee has {int(feature_value)} total working years."
                suggestion_text = "Invest in training and development to enhance skills and career prospects."
            else:
                reason_text = f"More Total Working Years: The employee has {int(feature_value)} total working years."
                suggestion_text = "Leverage their experience and provide opportunities for mentorship."
        elif feature == 'YearsInCurrentRole':
            if shap_value > 0:
                reason_text = f"Fewer Years in Current Role: The employee has been in their current role for {int(feature_value)} years."
                suggestion_text = "Provide opportunities for role enrichment or advancement."
            else:
                reason_text = f"More Years in Current Role: The employee has been in their current role for {int(feature_value)} years."
                suggestion_text = "Recognize their stability and offer new challenges."
        elif feature == 'YearsWithCurrManager':
            if shap_value > 0:
                reason_text = f"Fewer Years with Current Manager: The employee has been with their current manager for {int(feature_value)} years."
                suggestion_text = "Facilitate strong manager-employee relationships and provide leadership training."
            else:
                reason_text = f"More Years with Current Manager: The employee has been with their current manager for {int(feature_value)} years."
                suggestion_text = "Support effective management and team cohesion."
        elif feature == 'JobInvolvement':
            if shap_value > 0:
                reason_text = f"Low Job Involvement: The employee's job involvement is rated {int(feature_value)} (out of 4)."
                suggestion_text = "Increase opportunities for participation and feedback in decision-making."
            else:
                reason_text = f"High Job Involvement: The employee's job involvement is rated {int(feature_value)} (out of 4)."
                suggestion_text = "Continue to empower employees and value their input."
        elif feature.startswith('JobRole_'):
            role = feature.replace('JobRole_', '')
            if shap_value > 0:
                reason_text = f"Job Role: The '{role}' role might be associated with higher attrition risk."
                suggestion_text = f"Investigate specific challenges or stressors within the '{role}' role."
            else:
                reason_text = f"Job Role: The '{role}' role is associated with lower attrition risk."
                suggestion_text = f"Understand and replicate positive aspects of the '{role}' role."
        elif feature.startswith('MaritalStatus_'):
            status = feature.replace('MaritalStatus_', '')
            if shap_value > 0:
                reason_text = f"Marital Status: Being '{status}' might be a factor."
                suggestion_text = "Consider offering family-friendly policies or support programs."
            else:
                reason_text = f"Marital Status: Being '{status}' is a stabilizing factor."
                suggestion_text = "Support diverse employee needs and life stages."
        elif feature.startswith('BusinessTravel_'):
            travel = feature.replace('BusinessTravel_', '').replace('_', ' ')
            if shap_value > 0:
                reason_text = f"Business Travel: '{travel}' might contribute to attrition."
                suggestion_text = "Review travel policies and explore ways to reduce travel burden."
            else:
                reason_text = f"Business Travel: '{travel}' is not a significant attrition factor."
                suggestion_text = "Maintain current travel policies if they are effective."
        elif feature.startswith('EducationField_'):
            field = feature.replace('EducationField_', '').replace('_', ' ')
            if shap_value > 0:
                reason_text = f"Education Field: Employees in '{field}' might have higher attrition."
                suggestion_text = "Understand career aspirations and provide growth opportunities for employees in this field."
            else:
                reason_text = f"Education Field: Employees in '{field}' have lower attrition."
                suggestion_text = "Leverage their skills and provide relevant development."

        if reason_text and suggestion_text:
            reasons.append(reason_text)
            suggestions.append(suggestion_text)
            
    return reasons, suggestions


# ============================
# Routes
# ============================

# Landing page
@app.route('/')
def home():
    return render_template("home.html")


# Prediction page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template("index.html")
    try:
        form = request.form
        result, probability, reasons, suggestions = predict_attrition(
            float(form['Age']), float(form['DistanceFromHome']),
            float(form['EnvironmentSatisfaction']), float(form['JobInvolvement']),
            float(form['JobLevel']), float(form['JobSatisfaction']),
            float(form['MonthlyIncome']), float(form['StockOptionLevel']),
            float(form['TotalWorkingYears']), float(form['WorkLifeBalance']),
            float(form['YearsAtCompany']), float(form['YearsInCurrentRole']),
            float(form['YearsWithCurrManager']), form['OverTime'],
            form['JobRole'], form['MaritalStatus'], form['BusinessTravel'],
            form['EducationField']
        )
        return render_template("result.html", prediction=result, probability=f"{probability:.2f}", 
                             reasons=reasons, suggestions=suggestions)
    except Exception as e:
        return render_template("result.html", prediction=f"Error: {str(e)}", probability="")


# Dashboard route
@app.route('/dashboard')
def dashboard():
    try:
        # Load dataset
        df = pd.read_csv("HR-Employee-Attrition.csv")

        # --- Visualization 1: Overall Attrition ---
        fig1 = px.pie(df, names='Attrition', title='Overall Attrition Rate',
                      color_discrete_sequence=px.colors.sequential.RdBu)

        # --- Visualization 2: Attrition by Department ---
        dept_counts = df.groupby(['Department', 'Attrition']).size().reset_index(name='Count')
        fig2 = px.bar(dept_counts, x='Department', y='Count', color='Attrition',
                      barmode='group', title='Attrition by Department')

        # --- Visualization 3: Attrition by Gender ---
        gender_counts = df.groupby(['Gender', 'Attrition']).size().reset_index(name='Count')
        fig3 = px.bar(gender_counts, x='Gender', y='Count', color='Attrition',
                      barmode='group', title='Attrition by Gender')

        # --- Visualization 4: Attrition by Job Role ---
        jobrole_counts = df.groupby(['JobRole', 'Attrition']).size().reset_index(name='Count')
        fig4 = px.bar(jobrole_counts, x='Count', y='JobRole', color='Attrition',
                      orientation='h', title='Attrition by Job Role')

        # --- Visualization 5: Age vs Attrition ---
        fig5 = px.box(df, x='Attrition', y='Age', color='Attrition', title='Attrition vs Age')

        # --- Visualization 6: Monthly Income vs Attrition ---
        income_mean = df.groupby('Attrition')['MonthlyIncome'].mean().reset_index()
        fig6 = px.bar(income_mean, x='Attrition', y='MonthlyIncome', color='Attrition',title='Average Monthly Income by Attrition',text='MonthlyIncome')
        fig6.update_traces(texttemplate='%{text:.0f}', textposition='outside')
        
        
        # Convert to HTML
        graphs = [
            fig1.to_html(full_html=False),
            fig2.to_html(full_html=False),
            fig3.to_html(full_html=False),
            fig4.to_html(full_html=False),
            fig5.to_html(full_html=False),
            fig6.to_html(full_html=False)
        ]

        return render_template("dashboard.html", graphs=graphs)
    except Exception as e:
        return f"Error generating dashboard: {str(e)}"


# ============================
# Run App
# ============================

if __name__ == "__main__":
    app.run(debug=True)
