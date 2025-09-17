# ====================================
# Crop Production Prediction - Streamlit App
# ====================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================
# Page Config
# ==========================
st.set_page_config(
    page_title="🌾 Crop Production Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# Load Data
# ==========================
@st.cache_data
def load_data():
    df = pd.read_excel("FAOSTAT_data.xlsx")
    df = df[['Area', 'Item', 'Year', 'Element', 'Unit', 'Value']]
    df_pivot = df.pivot_table(values='Value',
                              index=['Area','Item','Year'],
                              columns='Element',
                              aggfunc='first').reset_index()

    df_pivot.rename(columns={
        "Area harvested": "Area_harvested",
        "Yield": "Yield",
        "Production": "Production"
    }, inplace=True)
    return df_pivot

df = load_data()

# ==========================
# Sidebar - User Chooses
# ==========================
st.sidebar.title("⚙️ User Options")
user_choice = st.sidebar.radio("Choose what you want to do:",
                               ["📊 Exploratory Data Analysis (EDA)",
                                "🤖 Modeling",
                                "🎯 Prediction"])

st.title("🌍 Crop Production Prediction Dashboard")

# ==========================
# Exploratory Data Analysis (EDA)
# ==========================
if user_choice == "📊 Exploratory Data Analysis (EDA)":
    st.header("📊 Exploratory Data Analysis")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(20))
    with col2:
        st.metric("Total Records", f"{len(df):,}")
        st.metric("Unique Crops", df['Item'].nunique())
        st.metric("Countries/Regions", df['Area'].nunique())

    st.markdown("---")
    st.subheader("Top 10 Crops by Records")
    crop_counts = df['Item'].value_counts().head(10)
    st.bar_chart(crop_counts)

    st.subheader("Production Trend Over Years")
    prod_trend = df.groupby("Year")["Production"].sum().reset_index()
    st.line_chart(prod_trend.set_index("Year"))

    st.subheader("Correlation Heatmap")
    corr = df[['Area_harvested','Yield','Production']].corr()
    fig, ax = plt.subplots(figsize=(5,3))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ==========================
# Modeling
# ==========================
elif user_choice == "🤖 Modeling":
    st.header("🤖 Model Training & Evaluation")

    X = df[['Area_harvested','Yield','Year']].fillna(0)
    y = df['Production'].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "R²": round(r2_score(y_test, y_pred), 3),
            "MAE": round(mean_absolute_error(y_test, y_pred), 2),
            "MSE": round(mean_squared_error(y_test, y_pred), 2)
        }

    results_df = pd.DataFrame(results).T
    st.subheader("📈 Model Performance")
    st.dataframe(results_df.style.highlight_max(axis=0, color="lightgreen"))

# ==========================
# Prediction (Dropdown for Year)
# ==========================
elif user_choice == "🎯 Prediction":
    st.header("🎯 Predict Total Crop Production (in tons)")

    col1, col2, col3 = st.columns(3)
    with col1:
        area = st.selectbox("🌍 Select Region (Area):", df['Area'].unique())
    with col2:
        crop = st.selectbox("🌱 Select Crop:", df['Item'].unique())
    with col3:
        year = st.selectbox("📅 Select Year:", sorted(df['Year'].unique()))

    st.markdown("---")

    # Train model
    X = df[['Area_harvested','Yield','Year']].fillna(0)
    y = df['Production'].fillna(0)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    # Get input data for selected values
    sample = df[(df['Area']==area) & (df['Item']==crop) & (df['Year']==year)]

    if sample.empty:
        st.warning("⚠️ No record found for this selection. Try another combination.")
    else:
        X_sample = sample[['Area_harvested','Yield','Year']].fillna(0)
        pred = model.predict(X_sample)

        st.success(f"✅ Predicted **Total Production** of **{crop}** in **{area} ({year})** "
                   f"is **{pred[0]:,.2f} tons**")

        if sample['Production'].notnull().values.any():
            actual = sample['Production'].values[0]
            st.info(f"📌 Actual Reported Production: {actual:,.2f} tons")

            fig, ax = plt.subplots()
            ax.bar(["Actual", "Predicted"], [actual, pred[0]], color=["skyblue", "orange"])
            ax.set_ylabel("Production (tons)")
            st.pyplot(fig)
