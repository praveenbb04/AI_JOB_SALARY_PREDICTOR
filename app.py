import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

px.defaults.template = "plotly_white"

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score

st.set_page_config(
    page_title="NaviBayes - AI Job Market Analytics",
    page_icon="🤖",
    layout="wide"
)

# --- UTILITY FUNCTIONS ---

@st.cache_data
def load_and_preprocess_data(file_path_or_buffer):
    """
    Loads raw AI Job Market dataset, extracts minimum/maximum/average salaries,
    and returns both the raw and preprocessed dataframes.
    """
    df = pd.read_csv(file_path_or_buffer)
    df_processed = df.copy()

    # Extract minimum and maximum salary from string format (e.g., '92860-109598')
    if "salary_range_usd" in df_processed.columns:
        df_processed["salary_min"] = (
            df_processed["salary_range_usd"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.split("-")
            .str[0]
            .astype(float)
        )
        df_processed["salary_max"] = (
            df_processed["salary_range_usd"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.split("-")
            .str[1]
            .astype(float)
        )
        df_processed["salary_avg"] = (df_processed["salary_min"] + df_processed["salary_max"]) / 2

    return df, df_processed


def get_scaled_numerical_data(df_processed):
    """
    Extracts and standardizes numeric columns for training/evaluation.
    """
    numeric_df = df_processed.select_dtypes(include=[np.number]).copy()
    
    # Remove index/identifier columns from features if present
    if "job_id" in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=["job_id"])

    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(numeric_df)
    scaled_df = pd.DataFrame(scaled_array, columns=numeric_df.columns)
    
    return numeric_df, scaled_df, scaler


# --- SIDEBAR CONFIGURATION ---

st.sidebar.title("🤖 AI Job Market")
uploaded_file = st.sidebar.file_uploader("Upload 'ai_job_market.csv'", type=["csv"])

# Target variable selection
if uploaded_file is not None:
    df_raw, df_processed = load_and_preprocess_data(uploaded_file)
else:
    st.sidebar.info("Please upload `ai_job_market.csv` to activate model training.")
    df_raw, df_processed = None, None

# --- MAIN APP LAYOUT ---

st.title("📊 AI Job Market Analytics & ML Lab")
st.markdown("Dynamic web application built from `NaviBayes.ipynb` workflow.")

tabs = st.tabs(["📋 Data Overview", "📈 EDA & Insights", "⚙️ Model Training & Evaluation", "🔮 Real-Time Prediction"])

# --- TAB 1: DATA OVERVIEW ---
with tabs[0]:
    st.header("Dataset Overview")
    if df_raw is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", df_raw.shape[0])
        col2.metric("Total Features", df_raw.shape[1])
        col3.metric("Processed Salary Features", 3)

        st.subheader("Raw Data Preview")
        st.dataframe(df_raw.head(10), use_container_width=True)

        st.subheader("Extracted Salary Features (Unscaled)")
        st.dataframe(df_processed[["salary_range_usd", "salary_min", "salary_max", "salary_avg"]].head(10), use_container_width=True)
    else:
        st.warning("Upload dataset file in the sidebar to view raw data.")

# --- TAB 2: EDA & INSIGHTS ---
with tabs[1]:
    st.header("Exploratory Data Analysis")
    if df_processed is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Salary Distribution by Industry")
            fig1 = px.box(df_processed, x="industry", y="salary_avg", color="industry", title="Average Salary ($) per Industry")
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.subheader("Experience Level Breakdown")
            fig2 = px.histogram(df_processed, x="experience_level", color="employment_type", barmode="group", title="Job Postings by Experience & Type")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Upload dataset file in the sidebar to unlock visualizations.")

# --- TAB 3: MODEL TRAINING & EVALUATION ---
with tabs[2]:
    st.header("Model Pipeline & Performance")
    if df_processed is not None:
        numeric_df, scaled_df, scaler = get_scaled_numerical_data(df_processed)

        st.sidebar.subheader("Model Configuration")
        target_col = st.sidebar.selectbox("Select Target Variable (Y)", numeric_df.columns, index=len(numeric_df.columns)-1)
        test_size = st.sidebar.slider("Test Split Ratio", 0.1, 0.4, 0.2, 0.05)
        
        feature_cols = [c for c in numeric_df.columns if c != target_col]

        X = scaled_df[feature_cols].values
        y = numeric_df[target_col].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        model_type = st.radio("Choose Model Task:", ["Linear Regression (Continuous Target)", "Logistic Regression (Categorical Target)"])

        if model_type == "Linear Regression (Continuous Target)":
            lin_reg = LinearRegression()
            lin_reg.fit(X_train, y_train)
            preds = lin_reg.predict(X_test)

            mse = mean_squared_error(y_test, preds)
            r2 = r2_score(y_test, preds)

            c1, c2 = st.columns(2)
            c1.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
            c2.metric("R² Score (%)", f"{r2 * 100:.2f}%")

            # Plot Actual vs Predicted
            fig = px.scatter(x=y_test, y=preds, labels={'x': 'Actual Values', 'y': 'Predicted Values'}, title="Actual vs. Predicted Target Values")
            st.plotly_chart(fig, use_container_width=True)

        else:
            max_iter = st.sidebar.slider("Logistic Regression Max Iterations", 100, 2000, 1000)
            
            # Categorize target variable for Logistic Regression
            y_train_cat = y_train.astype(int)
            y_test_cat = y_test.astype(int)

            clf = LogisticRegression(max_iter=max_iter)
            clf.fit(X_train, y_train_cat)
            preds = clf.predict(X_test)

            acc = accuracy_score(y_test_cat, preds)
            st.metric("Model Accuracy", f"{acc * 100:.2f}%")

            st.subheader("Classification Report")
            report_dict = classification_report(y_test_cat, preds, output_dict=True)
            st.dataframe(pd.DataFrame(report_dict).transpose(), use_container_width=True)
    else:
        st.warning("Upload dataset file to execute training pipeline.")

# --- TAB 4: REAL-TIME PREDICTION ---
with tabs[3]:
    st.header("Make Interactive Predictions")
    if df_processed is not None:
        st.subheader("Input Feature Values")
        
        numeric_df, scaled_df, scaler = get_scaled_numerical_data(df_processed)
        target_col = st.selectbox("Target Variable to Predict", numeric_df.columns, key="pred_target")
        feature_cols = [c for c in numeric_df.columns if c != target_col]

        input_data = {}
        cols = st.columns(len(feature_cols))
        
        for i, col in enumerate(feature_cols):
            min_val = float(numeric_df[col].min())
            max_val = float(numeric_df[col].max())
            mean_val = float(numeric_df[col].mean())
            input_data[col] = cols[i].number_input(f"{col}", min_value=min_val, max_value=max_val, value=mean_val)

        if st.button("Run Model Prediction"):
            # Prepare full feature array matching scaler input dimensions
            input_df = pd.DataFrame([input_data])
            
            # Model Training on current setup
            X = scaled_df[feature_cols].values
            y = numeric_df[target_col].values
            
            model = LinearRegression()
            model.fit(X, y)

            # Manual standardization using mean/scale of training set
            scaled_input = []
            for col in feature_cols:
                col_idx = list(numeric_df.columns).index(col)
                val_scaled = (input_data[col] - scaler.mean_[col_idx]) / scaler.scale_[col_idx]
                scaled_input.append(val_scaled)

            prediction = model.predict([scaled_input])[0]
            st.success(f"Predicted **{target_col}**: `${prediction:,.2f}`" if "salary" in target_col else f"Predicted **{target_col}**: `{prediction:.4f}`")
    else:
        st.warning("Upload dataset file in sidebar to run custom predictions.")