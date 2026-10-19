
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Sentinel | Churn Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"
DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


# =========================================================
# 2. LOAD MODEL AND DATASET
# =========================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. "
            "Run python src/train_model.py first."
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. "
            "Check that the CSV file is inside the data folder."
        )

    data = pd.read_csv(DATA_PATH)

    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"], errors="coerce"
    )

    data = data.dropna(subset=["TotalCharges", "Churn"])
    return data


try:
    model = load_model()
    df = load_data()
except Exception as error:
    st.error("Sentinel could not load the model or dataset.")
    st.code(str(error))
    st.stop()


# =========================================================
# 3. PREMIUM DARK THEME
# =========================================================

st.markdown("""
<style>
/* Overall background */
.stApp {
    background:
        radial-gradient(ellipse at 0% 0%, #182c48 0%, transparent 42%),
        linear-gradient(135deg, #080e1a, #0d1728 55%, #101b2d);
    color: #edf3fc;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.8rem;
    padding-bottom: 2rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101b2d, #0b1321);
    border-right: 1px solid #26384f;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #f3f7ff;
}

[data-testid="stSidebar"] label {
    color: #c5d3e5;
}

/* Hero section */
.hero {
    position: relative;
    overflow: hidden;
    padding: 32px;
    border-radius: 22px;
    background:
        radial-gradient(circle at 90% 5%, #285478 0%, transparent 38%),
        linear-gradient(115deg, #172943, #142b43 58%, #16394a);
    border: 1px solid #35516c;
    box-shadow: 0 14px 38px rgba(0, 0, 0, 0.20);
    margin-bottom: 24px;
}

.hero h1 {
    color: #ffffff;
    font-size: clamp(28px, 3vw, 40px);
    font-weight: 800;
    letter-spacing: -1.2px;
    margin: 8px 0;
}

.hero p {
    color: #c7d8eb;
    font-size: 15px;
    line-height: 1.7;
    margin-bottom: 0;
}

.eyebrow {
    color: #67e8f9;
    font-size: 11px;
    font-weight: 750;
    letter-spacing: 2.5px;
    text-transform: uppercase;
}

/* Metrics */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #17263a, #111e30);
    border: 1px solid #2a3d55;
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0 8px 22px rgba(0, 0, 0, 0.12);
    transition: border-color 0.2s ease;
}

[data-testid="stMetric"]:hover {
    border-color: #4b789d;
}

[data-testid="stMetricLabel"] {
    color: #a8bbd2 !important;
    font-size: 13px !important;
}

[data-testid="stMetricValue"] {
    color: #f4f8ff !important;
    font-size: clamp(22px, 2.2vw, 31px) !important;
    font-weight: 750 !important;
}

/* Headings and text */
h1, h2, h3 {
    color: #f3f7ff;
    letter-spacing: -0.4px;
}

[data-testid="stMarkdownContainer"] p {
    line-height: 1.65;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #b7c6d9;
    font-weight: 650;
    padding: 12px 16px;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #67e8f9;
}

/* Buttons */
div.stButton > button {
    min-height: 45px;
    border-radius: 11px;
    border: 1px solid #365775;
    background: linear-gradient(110deg, #147da1, #2563a6);
    color: white;
    font-weight: 650;
    transition: all 0.2s ease;
}

div.stButton > button:hover {
    border-color: #67e8f9;
    background: linear-gradient(110deg, #1596b9, #3477c2);
    color: white;
}

/* Input controls */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
    background-color: #111e30;
    border-color: #354960;
    border-radius: 9px;
}

/* Charts */
div[data-testid="stPlotlyChart"] {
    background: linear-gradient(145deg, #111d2e, #0e1929);
    border: 1px solid #263a51;
    border-radius: 16px;
    padding: 8px;
    margin-top: 8px;
    margin-bottom: 16px;
}

/* Tables and expanders */
[data-testid="stDataFrame"] {
    border: 1px solid #2a3d55;
    border-radius: 12px;
    overflow: hidden;
}

[data-testid="stExpander"] {
    border: 1px solid #2a3d55;
    border-radius: 12px;
    background: rgba(17, 30, 48, 0.65);
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Download button */
div[data-testid="stDownloadButton"] button {
    border-radius: 10px;
    border: 1px solid #365775;
    background: #162a40;
    color: #d9f4ff;
    font-weight: 650;
    min-height: 43px;
}

hr {
    border-color: #293a50;
}

[data-testid="stCaptionContainer"] {
    color: #93a8c0;
}

/* Smaller screens */
@media (max-width: 768px) {
    .block-container {
        padding: 1rem 0.8rem;
    }

    .hero {
        padding: 22px 18px;
        border-radius: 16px;
    }

    [data-testid="stMetric"] {
        padding: 12px;
    }
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 4. HELPER FUNCTIONS
# =========================================================

def apply_chart_theme(fig, height=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Arial, sans-serif",
            color="#dbe7f5",
            size=12
        ),
        margin=dict(l=20, r=20, t=65, b=25),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1")
        ),
        hoverlabel=dict(
            bgcolor="#17263a",
            font_color="#ffffff"
        )
    )

    if height is not None:
        fig.update_layout(height=height)

    return fig


# =========================================================
# 5. SIDEBAR — CUSTOMER PROFILE
# =========================================================

st.sidebar.markdown("# 🛡️ Sentinel")
st.sidebar.caption("CUSTOMER INTELLIGENCE PLATFORM")
st.sidebar.divider()

st.sidebar.subheader("Customer Profile")
st.sidebar.caption(
    "Adjust the customer details and run an analysis."
)

with st.sidebar:
    with st.form("customer_form"):

        st.markdown("#### 👤 Personal Details")

        gender = st.selectbox("Gender", ["Female", "Male"])

        senior = st.selectbox(
            "Senior citizen", ["No", "Yes"]
        )

        partner = st.selectbox(
            "Has partner?", ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Has dependents?", ["Yes", "No"]
        )

        tenure = st.slider(
            "Customer tenure (months)", 0, 72, 12
        )

        st.divider()
        st.markdown("#### 📡 Services")

        phone = st.selectbox(
            "Phone service", ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple lines",
            ["No", "Yes", "No phone service"]
        )

        internet = st.selectbox(
            "Internet service",
            ["DSL", "Fiber optic", "No"]
        )

        security = st.selectbox(
            "Online security",
            ["Yes", "No", "No internet service"]
        )

        backup = st.selectbox(
            "Online backup",
            ["Yes", "No", "No internet service"]
        )

        protection = st.selectbox(
            "Device protection",
            ["Yes", "No", "No internet service"]
        )

        support = st.selectbox(
            "Tech support",
            ["Yes", "No", "No internet service"]
        )

        tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        movies = st.selectbox(
            "Streaming movies",
            ["Yes", "No", "No internet service"]
        )

        st.divider()
        st.markdown("#### 💳 Billing & Contract")

        contract = st.selectbox(
            "Contract type",
            ["Month-to-month", "One year", "Two year"]
        )

        billing = st.selectbox(
            "Paperless billing", ["Yes", "No"]
        )

        payment = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly = st.number_input(
            "Monthly charges ($)",
            min_value=0.0,
            max_value=1000.0,
            value=70.0,
            step=1.0
        )

        total = st.number_input(
            "Total charges ($)",
            min_value=0.0,
            max_value=100000.0,
            value=840.0,
            step=10.0
        )

        run_prediction = st.form_submit_button(
            "🔍 Analyze Customer",
            type="primary",
            use_container_width=True
        )


# =========================================================
# 6. BUILD CUSTOMER DATAFRAME
# =========================================================

customer = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multiple_lines,
    "InternetService": internet,
    "OnlineSecurity": security,
    "OnlineBackup": backup,
    "DeviceProtection": protection,
    "TechSupport": support,
    "StreamingTV": tv,
    "StreamingMovies": movies,
    "Contract": contract,
    "PaperlessBilling": billing,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": total
}])


# =========================================================
# 7. RUN AND STORE PREDICTION
# =========================================================

if run_prediction:
    try:
        prediction = int(model.predict(customer)[0])
        probabilities = model.predict_proba(customer)[0]
        churn_probability = float(probabilities[1]) * 100

        st.session_state["prediction"] = prediction
        st.session_state["churn_probability"] = churn_probability
        st.session_state["last_customer"] = customer.copy()

        st.session_state["prediction_error"] = None

    except Exception as error:
        st.session_state["prediction_error"] = str(error)


# =========================================================
# 8. PREMIUM HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <div class="eyebrow">AI-powered customer analytics</div>
    <h1>Sentinel Churn Detector</h1>
    <p>
        Understand customer behavior. Estimate churn risk.
        Turn historical patterns into informed decisions.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 9. HISTORICAL DATASET KPI CARDS
# =========================================================

total_customers = len(df)
churned_customers = int((df["Churn"] == "Yes").sum())
retained_customers = int((df["Churn"] == "No").sum())
overall_churn = (df["Churn"] == "Yes").mean() * 100
avg_monthly = df["MonthlyCharges"].mean()

st.markdown("### 📌 Dataset at a Glance")

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Total Customers",
    f"{total_customers:,}"
)

k2.metric(
    "Churned Customers",
    f"{churned_customers:,}"
)

k3.metric(
    "Historical Churn Rate",
    f"{overall_churn:.1f}%"
)

k4.metric(
    "Avg. Monthly Charges",
    f"${avg_monthly:.2f}"
)

st.caption(
    "These KPIs summarize the historical dataset, not live customer activity."
)


# =========================================================
# 10. NAVIGATION TABS
# =========================================================

overview_tab, prediction_tab, insights_tab = st.tabs([
    "📊 Overview",
    "🎯 Customer Prediction",
    "📈 Explore Insights"
])


# =========================================================
# 11. OVERVIEW TAB
# =========================================================

with overview_tab:

    st.subheader("Customer Churn Overview")

    st.write(
        "Explore the historical dataset to understand retention "
        "patterns and differences between customer groups."
    )

    left, right = st.columns(2)

    # Churn distribution
    with left:
        churn_counts = (
            df["Churn"]
            .value_counts()
            .rename_axis("Status")
            .reset_index(name="Customers")
        )

        churn_counts["Status"] = churn_counts["Status"].map({
            "No": "Retained",
            "Yes": "Churned"
        })

        fig = px.pie(
            churn_counts,
            names="Status",
            values="Customers",
            hole=0.68,
            title="Retention vs Churn",
            color="Status",
            color_discrete_map={
                "Retained": "#38bdf8",
                "Churned": "#fb7185"
            }
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent",
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Customers: %{value:,}<br>"
                "Share: %{percent}<extra></extra>"
            )
        )

        apply_chart_theme(fig, 370)
        st.plotly_chart(fig, use_container_width=True)

    # Contract churn rates
    with right:
        contract_rates = (
            df.groupby("Contract")["Churn"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .reset_index(name="Churn Rate")
            .sort_values("Churn Rate", ascending=True)
        )

        fig = px.bar(
            contract_rates,
            x="Churn Rate",
            y="Contract",
            orientation="h",
            title="Churn Rate by Contract",
            text=contract_rates["Churn Rate"].map(
                lambda value: f"{value:.1f}%"
            ),
            color="Churn Rate",
            color_continuous_scale=[
                "#38bdf8", "#fbbf24", "#fb7185"
            ],
            labels={
                "Churn Rate": "Churn rate (%)",
                "Contract": "Contract type"
            }
        )

        fig.update_traces(
            textposition="outside",
            cliponaxis=False
        )

        fig.update_layout(
            coloraxis_showscale=False,
            xaxis_range=[
                0,
                min(110, float(contract_rates["Churn Rate"].max()) + 15)
            ]
        )

        apply_chart_theme(fig, 370)
        st.plotly_chart(fig, use_container_width=True)

    # Monthly charges distribution
    st.subheader("💰 Monthly Charges Analysis")

    fig = px.histogram(
        df,
        x="MonthlyCharges",
        color="Churn",
        nbins=35,
        barmode="overlay",
        opacity=0.78,
        title="Monthly Charges Distribution by Churn Status",
        color_discrete_map={
            "No": "#38bdf8",
            "Yes": "#fb7185"
        },
        labels={
            "MonthlyCharges": "Monthly charges ($)",
            "count": "Number of customers",
            "Churn": "Churned?"
        }
    )

    apply_chart_theme(fig, 390)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ How to read these charts"):
        st.write(
            "- The donut chart shows the proportion of customers "
            "who stayed and those who left.\n"
            "- The contract chart compares historical churn rates "
            "for different contract types.\n"
            "- The histogram compares monthly charge distributions "
            "for customers who stayed versus those who left."
        )


# =========================================================
# 12. CUSTOMER PREDICTION TAB
# =========================================================

with prediction_tab:

    st.subheader("Individual Customer Risk Assessment")

    st.write(
        "Use the customer profile in the sidebar and click "
        "**Analyze Customer** to generate a prediction."
    )

    if st.session_state.get("prediction_error"):
        st.error("Prediction could not be completed.")
        st.code(st.session_state["prediction_error"])

    elif "prediction" not in st.session_state:

        st.info(
            "👈 Start by entering customer details in the sidebar, "
            "then click **Analyze Customer**."
        )

        st.markdown("#### What you'll receive")

        c1, c2, c3 = st.columns(3)

        c1.markdown("### 🎯")
        c1.markdown("**Churn probability**")
        c1.caption("The model's estimated probability of churn.")

        c2.markdown("### 📊")
        c2.markdown("**Risk visualization**")
        c2.caption("A gauge showing the estimated probability.")

        c3.markdown("### 💡")
        c3.markdown("**Suggested action**")
        c3.caption("A general recommendation based on the risk band.")

    else:

        prediction = st.session_state["prediction"]
        probability = st.session_state["churn_probability"]
        last_customer = st.session_state["last_customer"]

        # Prediction summary
        if probability >= 60:
            risk_label = "Higher estimated risk"
            risk_color = "#fb7185"
        elif probability >= 30:
            risk_label = "Moderate estimated risk"
            risk_color = "#fbbf24"
        else:
            risk_label = "Lower estimated risk"
            risk_color = "#34d399"

        st.markdown(
            f"""
            <div style="
                padding: 22px;
                border-radius: 16px;
                background: #132238;
                border: 1px solid {risk_color};
                margin: 12px 0 22px 0;
            ">
                <div style="
                    font-size: 12px;
                    letter-spacing: 2px;
                    color: #a8bbd2;
                    text-transform: uppercase;
                ">Prediction status</div>
                <div style="
                    font-size: 25px;
                    font-weight: 750;
                    color: {risk_color};
                    margin: 8px 0;
                ">{risk_label}</div>
                <div style="color: #c7d8eb;">
                    {"The model predicts this customer will churn."
                    if prediction == 1
                    else "The model predicts this customer will stay."}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        p1, p2, p3 = st.columns(3)

        p1.metric(
            "Estimated Churn Probability",
            f"{probability:.1f}%"
        )

        p2.metric(
            "Predicted Outcome",
            "Likely to Churn" if prediction == 1 else "Likely to Stay"
        )

        p3.metric(
            "Customer Tenure",
            f"{int(last_customer['tenure'].iloc[0])} months"
        )

        # Probability gauge
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability,
                number={
                    "suffix": "%",
                    "font": {"color": "#f3f7ff", "size": 36}
                },
                title={
                    "text": "Estimated Churn Probability",
                    "font": {"color": "#cbd5e1", "size": 17}
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "ticksuffix": "%",
                        "tickcolor": "#a8bbd2"
                    },
                    "bar": {
                        "color": risk_color,
                        "thickness": 0.28
                    },
                    "bgcolor": "#101b2d",
                    "borderwidth": 1,
                    "bordercolor": "#2a3d55",
                    "steps": [
                        {"range": [0, 30], "color": "#173a35"},
                        {"range": [30, 60], "color": "#49391b"},
                        {"range": [60, 100], "color": "#4a2533"}
                    ]
                }
            )
        )

        apply_chart_theme(gauge, 320)
        gauge.update_layout(margin=dict(l=35, r=35, t=75, b=20))
        st.plotly_chart(gauge, use_container_width=True)

        # Suggested actions
        st.subheader("💡 Suggested Next Steps")

        if probability >= 60:
            st.error(
                "Higher estimated risk: consider a customer check-in."
            )
            st.write(
                "- Review customer concerns or support history.\n"
                "- Check whether the current service plan meets their needs.\n"
                "- Consider appropriate retention options."
            )

        elif probability >= 30:
            st.warning(
                "Moderate estimated risk: monitor this customer."
            )
            st.write(
                "- Review customer satisfaction.\n"
                "- Check billing or service questions.\n"
                "- Consider a routine customer check-in."
            )

        else:
            st.success(
                "Lower estimated risk according to this model."
            )
            st.write(
                "- Continue normal customer engagement.\n"
                "- Maintain service quality.\n"
                "- Reassess if the customer's circumstances change."
            )

        # Customer profile
        with st.expander("👤 View analyzed customer profile"):
            st.dataframe(
                last_customer,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇️ Download this customer profile",
                data=last_customer.to_csv(index=False).encode("utf-8"),
                file_name="sentinel_customer_profile.csv",
                mime="text/csv"
            )

        st.caption(
            "The probability is a model estimate, not a guarantee. "
            "Risk thresholds are illustrative and have not been "
            "validated as business decision rules."
        )


# =========================================================
# 13. EXPLORE INSIGHTS TAB
# =========================================================

with insights_tab:

    st.subheader("Explore Historical Customer Data")

    st.write(
        "Filter the historical dataset and explore customer "
        "tenure and churn patterns."
    )

    selected_contract = st.selectbox(
        "Filter by contract type",
        ["All contracts"] +
        sorted(df["Contract"].dropna().unique().tolist())
    )

    filtered_df = df.copy()

    if selected_contract != "All contracts":
        filtered_df = filtered_df[
            filtered_df["Contract"] == selected_contract
        ]

    total_in_view = len(filtered_df)

    if total_in_view:
        churn_rate_in_view = (
            (filtered_df["Churn"] == "Yes").mean() * 100
        )
        average_tenure = filtered_df["tenure"].mean()
    else:
        churn_rate_in_view = 0
        average_tenure = 0

    i1, i2, i3 = st.columns(3)

    i1.metric(
        "Customers in View",
        f"{total_in_view:,}"
    )

    i2.metric(
        "Churn Rate in View",
        f"{churn_rate_in_view:.1f}%"
    )

    i3.metric(
        "Average Tenure",
        f"{average_tenure:.1f} months"
    )

    # Tenure distribution
    tenure_chart = px.histogram(
        filtered_df,
        x="tenure",
        color="Churn",
        nbins=24,
        barmode="group",
        title="Customer Tenure by Churn Status",
        color_discrete_map={
            "No": "#38bdf8",
            "Yes": "#fb7185"
        },
        labels={
            "tenure": "Tenure (months)",
            "count": "Customers",
            "Churn": "Churned?"
        }
    )

    apply_chart_theme(tenure_chart, 390)
    st.plotly_chart(tenure_chart, use_container_width=True)

    # Contract summary table
    st.subheader("📋 Contract Summary")

    if total_in_view:
        summary = (
            filtered_df.groupby("Contract")
            .agg(
                Customers=("Churn", "size"),
                Churned=("Churn", lambda x: (x == "Yes").sum()),
                AverageTenure=("tenure", "mean"),
                AverageMonthlyCharges=("MonthlyCharges", "mean")
            )
            .reset_index()
        )

        summary["ChurnRate"] = (
            summary["Churned"] / summary["Customers"] * 100
        )

        summary = summary.rename(columns={
            "AverageTenure": "Average Tenure (months)",
            "AverageMonthlyCharges": "Average Monthly Charges ($)",
            "ChurnRate": "Churn Rate (%)"
        })

        st.dataframe(
            summary.round(2),
            use_container_width=True,
            hide_index=True
        )

    # Customer records
    st.subheader("🔎 Customer Records")

    display_columns = [
        "gender",
        "tenure",
        "Contract",
        "MonthlyCharges",
        "TotalCharges",
        "Churn"
    ]

    st.dataframe(
        filtered_df[display_columns].head(100),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "The table displays up to the first 100 matching records."
    )

    # CSV export
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Filtered Dataset (CSV)",
        data=csv_data,
        file_name="sentinel_customer_data.csv",
        mime="text/csv",
        use_container_width=True
    )


# =========================================================
# 14. FOOTER
# =========================================================

st.divider()

footer_left, footer_right = st.columns(2)

with footer_left:
    st.caption(
        "🛡️ SENTINEL CHURN DETECTOR · Customer Analytics"
    )

with footer_right:
    st.caption(
        "Machine learning prototype · Historical data analysis"
    )
