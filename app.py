
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="AI Spending Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("🤖 AI Customer Spending Prediction")
st.caption("Mini Project 1 | Linear Regression | Machine Learning")

# Generate sample dataset
@st.cache_data
def generate_data():
    rng = np.random.default_rng(42)
    n = 500

    age = rng.integers(18, 65, n)
    income = rng.integers(200000, 1500000, n)
    family = rng.integers(1, 7, n)
    visits = rng.integers(0, 31, n)
    previous = rng.integers(500, 30000, n)

    spending = (
        1000
        + income / 1500
        + family * 500
        + visits * 100
        + previous * 0.25
        - age * 10
        + rng.normal(0, 1500, n)
    )

    return pd.DataFrame({
        "Age": age,
        "Annual_Income": income,
        "Family_Size": family,
        "Website_Visits": visits,
        "Previous_Spending": previous,
        "Monthly_Spending": np.maximum(spending, 500).round(2)
    })

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard",
        "Data Cleaning",
        "Model Training",
        "Prediction Studio"
    ]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV (Optional)",
    type=["csv"]
)

# Load data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = generate_data()

features = [
    "Age",
    "Annual_Income",
    "Family_Size",
    "Website_Visits",
    "Previous_Spending"
]
target = "Monthly_Spending"

required = features + [target]

missing_columns = [c for c in required if c not in df.columns]

if missing_columns:
    st.error("CSV missing columns: " + ", ".join(missing_columns))
    st.stop()

df = df[required].copy()

for col in required:
    df[col] = pd.to_numeric(df[col], errors="coerce")

raw_rows = len(df)
duplicates = int(df.duplicated().sum())
missing_values = int(df.isna().sum().sum())

clean_df = df.drop_duplicates().dropna(subset=[target])
clean_df = clean_df.dropna(subset=features, how="all")

if len(clean_df) < 10:
    st.error("At least 10 usable rows are required.")
    st.stop()

# Train Linear Regression model
X = clean_df[features]
y = clean_df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = make_pipeline(
    SimpleImputer(strategy="median"),
    LinearRegression()
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Dashboard
if page == "Dashboard":

    st.subheader("Executive Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Customers", f"{raw_rows:,}")
    c2.metric("Clean Records", f"{len(clean_df):,}")
    c3.metric("MAE", f"₹{mae:,.0f}")
    c4.metric("R² Score", f"{r2:.3f}")

    left, right = st.columns(2)

    with left:
        fig = px.histogram(
            clean_df,
            x=target,
            title="Monthly Spending Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.scatter(
            clean_df,
            x="Annual_Income",
            y=target,
            title="Annual Income vs Monthly Spending"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dataset Preview")
    st.dataframe(clean_df.head(20), use_container_width=True)

# Data Cleaning
elif page == "Data Cleaning":

    st.subheader("Data Cleaning Module")

    a, b, c = st.columns(3)
    a.metric("Original Rows", raw_rows)
    b.metric("Duplicates Found", duplicates)
    c.metric("Missing Values Found", missing_values)

    st.write("Missing Values by Column")
    st.dataframe(df.isna().sum().rename("Missing Values"))

    st.write("Cleaned Dataset")
    st.dataframe(clean_df, use_container_width=True)

    st.download_button(
        "Download Cleaned CSV",
        clean_df.to_csv(index=False),
        "cleaned_customer_data.csv",
        "text/csv"
    )

# Model Training
elif page == "Model Training":

    st.subheader("Linear Regression Model Training")

    a, b, c, d = st.columns(4)

    a.metric("MAE", f"₹{mae:,.2f}")
    b.metric("MSE", f"{mse:,.2f}")
    c.metric("RMSE", f"₹{rmse:,.2f}")
    d.metric("R² Score", f"{r2:.3f}")

    st.write(f"Training Records: {len(X_train)}")
    st.write(f"Testing Records: {len(X_test)}")

    results = pd.DataFrame({
        "Actual Spending": y_test.to_numpy(),
        "Predicted Spending": y_pred
    })

    fig = px.scatter(
        results,
        x="Actual Spending",
        y="Predicted Spending",
        title="Actual vs Predicted Values"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(results.round(2), use_container_width=True)

    st.download_button(
        "Download Predictions",
        results.to_csv(index=False),
        "linear_regression_results.csv",
        "text/csv"
    )

# Prediction Studio
elif page == "Prediction Studio":

    st.subheader("AI Prediction Studio")

    st.write("Enter customer details to predict monthly spending.")

    with st.form("prediction_form"):

        a, b = st.columns(2)

        with a:
            age = st.number_input(
                "Age", min_value=18, max_value=100, value=30
            )
            income = st.number_input(
                "Annual Income (₹)",
                min_value=0,
                value=600000,
                step=10000
            )
            family = st.number_input(
                "Family Size", min_value=1, max_value=20, value=4
            )

        with b:
            visits = st.number_input(
                "Website Visits per Month",
                min_value=0,
                max_value=1000,
                value=10
            )
            previous = st.number_input(
                "Previous Monthly Spending (₹)",
                min_value=0,
                value=8000,
                step=500
            )

        submit = st.form_submit_button("Predict Spending")

    if submit:

        customer = pd.DataFrame([{
            "Age": age,
            "Annual_Income": income,
            "Family_Size": family,
            "Website_Visits": visits,
            "Previous_Spending": previous
        }], columns=features)

        prediction = float(model.predict(customer)[0])
        prediction = max(0, prediction)

        st.success("Prediction completed successfully!")

        st.metric(
            "Estimated Monthly Spending",
            f"₹{prediction:,.2f}"
        )

        st.caption(
            "This is an educational estimate based on the trained model."
        )
