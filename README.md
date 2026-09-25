# 🤖 AI Job Market Analytics & Machine Learning

## Slide 1 — Title

### **AI Job Market Analytics Using Machine Learning**

**Project Type:** Machine Learning & Data Analytics
**Domain:** Artificial Intelligence / Job Market
**Technology:** Python, Pandas, NumPy, Scikit-learn, Streamlit, Plotly

### Project Overview

This project analyzes an **AI Job Market dataset** and applies Machine Learning techniques to understand numerical job-market data and perform predictions.

The project includes:

* 📊 Data preprocessing
* 🔢 Salary feature extraction
* 📏 Feature standardization
* 📈 Exploratory Data Analysis
* 🤖 Linear Regression
* 🧠 Logistic Regression
* 📋 Model evaluation
* 🔮 Interactive prediction using Streamlit

The Streamlit application is titled **“NaviBayes - AI Job Market Analytics”** and provides an interactive interface for analyzing and training models on the uploaded dataset.

---

# Slide 2 — Objective & Implementation Steps

## 🎯 Objective

The main objective of this project is to use Machine Learning techniques to analyze AI-related job-market data and demonstrate how preprocessing, standardization, regression, classification, and prediction can be implemented in a practical application.

### Key Objectives

* Analyze AI job-market information.
* Extract useful salary-related features.
* Standardize numerical data.
* Explore relationships between job attributes and salary.
* Train Machine Learning models.
* Evaluate model performance.
* Provide an interactive prediction interface.

The application allows users to upload `ai_job_market.csv` and activate the data-processing and ML pipeline.

## ⚙️ Implementation Steps

### Step 1 — Load Dataset

The AI Job Market dataset is loaded using Pandas.

```python
df = pd.read_csv("ai_job_market.csv")
```

### Step 2 — Data Preprocessing

Salary ranges such as:

```text
$92,860-$109,598
```

are separated into:

* `salary_min`
* `salary_max`
* `salary_avg`

The application performs this extraction automatically.

### Step 3 — Standardization

Numerical features are standardized using `StandardScaler`.

```python
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)
```

Standardization converts numerical features into a common scale.

### Step 4 — Train/Test Split

The dataset is divided into training and testing data.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

### Step 5 — Machine Learning

Two models are implemented:

**Linear Regression**

* Used for continuous numerical targets.
* Evaluated using MSE and R².

**Logistic Regression**

* Used for categorical/classification targets.
* Evaluated using accuracy and classification report.

The Streamlit application provides both model choices through the model-training interface.

### Step 6 — Prediction

The trained model receives user-provided feature values and generates a prediction.

The application also standardizes the input using the training-set mean and scale before prediction.

---

# Slide 3 — Code Screenshot

## 💻 Code Implementation

### Important Code Sections

#### 1. Salary Feature Extraction

```python
df["salary_min"] = (
    df["salary_range_usd"]
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.split("-")
    .str[0]
    .astype(float)
)

df["salary_max"] = (
    df["salary_range_usd"]
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.split("-")
    .str[1]
    .astype(float)
)

df["salary_avg"] = (
    df["salary_min"] + df["salary_max"]
) / 2
```

#### 2. Standardization

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df[["salary_min",
    "salary_max",
    "salary_avg"]] = scaler.fit_transform(
        df[["salary_min",
            "salary_max",
            "salary_avg"]]
)
```

#### 3. Linear Regression

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

#### 4. Model Evaluation

```python
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("MSE:", mse)
print("R2:", r2)
```

#### 5. Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

### 📸 Screenshot to Add

> **Insert Screenshot:** Jupyter Notebook showing the preprocessing, standardization, model training and evaluation code.

Suggested filename:

```text
images/code-screenshot.png
```

The Streamlit implementation imports `StandardScaler`, `LogisticRegression`, `LinearRegression`, and the required evaluation metrics.

---

# Slide 4 — Output Screenshots

## 📊 Project Outputs

The application produces several interactive outputs.

### 1. Dataset Overview

Displays:

* Total number of rows
* Total number of features
* Processed salary features
* Raw dataset preview
* Extracted salary features

The application displays these metrics and tables in the **Data Overview** tab.

### 2. Exploratory Data Analysis

The application generates visualizations such as:

**Salary Distribution by Industry**

```text
Industry → Salary Distribution
```

**Experience Level Breakdown**

```text
Experience Level → Job Postings
Employment Type → Comparison
```

These visualizations are created using Plotly.

### 3. Linear Regression Output

The model displays:

* Mean Squared Error (MSE)
* R² Score
* Actual vs Predicted visualization

```text
MSE       → Model Error
R² Score  → Model Fit
Graph     → Actual vs Predicted
```

### 4. Logistic Regression Output

The application displays:

* Model Accuracy
* Classification Report

```text
Accuracy
Precision
Recall
F1-Score
```

### 5. Interactive Prediction

Users can enter numerical feature values and click:

**“Run Model Prediction”**

The application then displays the predicted target value.

### 📸 Screenshots to Add

Add these screenshots to your presentation/GitHub:

```text
images/
├── dataset-output.png
├── eda-output.png
├── model-performance.png
└── prediction-output.png
```

Example Markdown:

```markdown
![Dataset Overview](images/dataset-output.png)

![EDA Output](images/eda-output.png)

![Model Performance](images/model-performance.png)

![Prediction Output](images/prediction-output.png)
```

---

# Slide 5 — Applications / Uses

## 🚀 Applications & Uses

This project demonstrates how Machine Learning can be applied to job-market analytics.

### 1. 💼 Job Market Analysis

Organizations can analyze job postings and understand patterns in:

* Industries
* Experience levels
* Employment types
* Salaries

### 2. 💰 Salary Analysis

The extracted salary features can help analyze:

* Minimum salary
* Maximum salary
* Average salary
* Salary distribution across industries

### 3. 📊 Data-Driven Insights

EDA can help identify patterns in job-market data and support data-driven analysis.

### 4. 🤖 Machine Learning Demonstration

The project provides a practical example of:

* Data preprocessing
* Feature scaling
* Regression
* Classification
* Model evaluation
* Prediction

### 5. 🎓 Educational Application

The project can be used by students to understand a complete basic ML workflow:

```text
Dataset
   ↓
Preprocessing
   ↓
Standardization
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Evaluation
   ↓
Prediction
```

### 6. 🌐 Interactive ML Application

The Streamlit interface converts the notebook-based ML workflow into an interactive web application where users can upload a dataset, explore data, train models, and perform predictions.

---

# Slide 6 — Conclusion & References

## ✅ Conclusion

The **AI Job Market Analytics** project demonstrates a complete beginner-friendly Machine Learning workflow using real-world job-market data.

The project successfully covers:

* 📥 Dataset loading
* 🧹 Data preprocessing
* 💰 Salary feature extraction
* 📏 Standardization
* 📊 Exploratory Data Analysis
* 🤖 Linear Regression
* 🧠 Logistic Regression
* 📈 Model evaluation
* 🔮 Interactive prediction

The Streamlit application brings these components together into a single interactive ML dashboard.

Overall, the project demonstrates how raw job-market data can be transformed into structured features, analyzed using visualization techniques, and used with Machine Learning models for prediction.

## 📚 References

1. **Python Documentation**
   https://docs.python.org/

2. **Pandas Documentation**
   https://pandas.pydata.org/docs/

3. **NumPy Documentation**
   https://numpy.org/doc/

4. **Scikit-learn Documentation**
   https://scikit-learn.org/

5. **Streamlit Documentation**
   https://docs.streamlit.io/

6. **Plotly Python Documentation**
   https://plotly.com/python/

### Project Files

```text
AI-Job-Market-Analytics/
│
├── app(1).py
├── pro.ipynb
├── ai_job_market.csv
├── README.md
│
└── images/
    ├── code-screenshot.png
    ├── dataset-output.png
    ├── eda-output.png
    ├── model-performance.png
    └── prediction-output.png
```
