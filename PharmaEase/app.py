from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.employee import predict_staff
from modules.inventory import get_available_drugs, predict_inventory
from modules.prescription import predict_drug
from modules.sales import predict_sales


st.set_page_config(
    page_title="PharmaEase Dashboard",
    page_icon="PE",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
BACKGROUND_IMAGE = PROJECT_ROOT / "background.jpg"

WEEKDAY_OPTIONS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


@st.cache_data
def load_sales_data():
    df = pd.read_csv(DATA_DIR / "pharmacy_sales_dataset.csv", parse_dates=["date"])
    df["revenue"] = df["quantity"] * df["price"]
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_name"] = df["date"].dt.day_name()
    return df


@st.cache_data
def load_employee_data():
    return pd.read_csv(DATA_DIR / "employee_management_dataset.csv")


@st.cache_data
def load_prescription_data():
    return pd.read_csv(DATA_DIR / "prescription_tracking_dataset.csv", parse_dates=["date"])


@st.cache_data
def load_inventory_data():
    return pd.read_csv(DATA_DIR / "inventory.csv", parse_dates=["date"])


def inject_styles():
    bg = BACKGROUND_IMAGE.as_posix()
    st.markdown(
        f"""
        <style>
        :root {{
            --bg: #f4efe6;
            --panel: rgba(255, 248, 240, 0.94);
            --panel-strong: rgba(255, 252, 247, 0.98);
            --text: #14232d;
            --muted: #425466;
            --accent: #0f766e;
            --accent-2: #f59e0b;
            --border: rgba(15, 118, 110, 0.2);
            --shadow: 0 18px 40px rgba(65, 54, 37, 0.12);
        }}

        .stApp {{
            background:
                linear-gradient(135deg, rgba(244, 239, 230, 0.94), rgba(253, 246, 236, 0.9)),
                url("{bg}") center/cover fixed no-repeat;
            color: var(--text);
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, rgba(16, 88, 84, 0.97), rgba(10, 53, 62, 0.98));
        }}

        [data-testid="stSidebar"] * {{
            color: #f8fafc;
        }}

        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}

        [data-testid="stAppViewContainer"] {{
            color: var(--text);
        }}

        [data-testid="stAppViewContainer"] p,
        [data-testid="stAppViewContainer"] label,
        [data-testid="stAppViewContainer"] span,
        [data-testid="stAppViewContainer"] li,
        [data-testid="stAppViewContainer"] .stMarkdown,
        [data-testid="stAppViewContainer"] .stCaption,
        [data-testid="stAppViewContainer"] .stText,
        [data-testid="stAppViewContainer"] .stSubheader,
        [data-testid="stAppViewContainer"] .stHeader {{
            color: var(--text);
        }}

        [data-testid="stAppViewContainer"] h1,
        [data-testid="stAppViewContainer"] h2,
        [data-testid="stAppViewContainer"] h3,
        [data-testid="stAppViewContainer"] h4 {{
            color: #10202a;
        }}

        [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] {{
            color: var(--muted);
        }}

        [data-testid="stAppViewContainer"] .stAlert {{
            color: var(--text);
        }}

        [data-testid="stAppViewContainer"] [data-baseweb="select"] > div,
        [data-testid="stAppViewContainer"] .stDateInput input,
        [data-testid="stAppViewContainer"] .stNumberInput input,
        [data-testid="stAppViewContainer"] .stTextArea textarea,
        [data-testid="stAppViewContainer"] .stTextInput input {{
            color: var(--text);
            background: rgba(255, 255, 255, 0.9);
        }}

        [data-testid="stAppViewContainer"] .stDataFrame,
        [data-testid="stAppViewContainer"] [data-testid="stTable"] {{
            color: var(--text);
        }}

        [data-testid="stAppViewContainer"] .stButton > button,
        [data-testid="stAppViewContainer"] .stDownloadButton > button,
        [data-testid="stAppViewContainer"] div[data-testid="stFormSubmitButton"] button,
        [data-testid="stAppViewContainer"] button[kind="primary"],
        [data-testid="stAppViewContainer"] button[kind="secondary"],
        [data-testid="stAppViewContainer"] button[data-testid="baseButton-secondary"],
        [data-testid="stAppViewContainer"] button[data-testid="baseButton-primary"] {{
            appearance: none !important;
            background: linear-gradient(135deg, #fef3c7, #fde68a) !important;
            background-color: #fde68a !important;
            color: #10202a !important;
            -webkit-text-fill-color: #10202a !important;
            border: 1px solid #d97706 !important;
            border-radius: 12px !important;
            box-shadow: 0 10px 22px rgba(217, 119, 6, 0.18) !important;
            font-weight: 700 !important;
            opacity: 1 !important;
            min-height: 2.75rem;
            padding: 0.45rem 1rem !important;
        }}

        [data-testid="stAppViewContainer"] .stButton > button:hover,
        [data-testid="stAppViewContainer"] .stDownloadButton > button:hover,
        [data-testid="stAppViewContainer"] div[data-testid="stFormSubmitButton"] button:hover,
        [data-testid="stAppViewContainer"] button[kind="primary"]:hover,
        [data-testid="stAppViewContainer"] button[kind="secondary"]:hover,
        [data-testid="stAppViewContainer"] button[data-testid="baseButton-secondary"]:hover,
        [data-testid="stAppViewContainer"] button[data-testid="baseButton-primary"]:hover {{
            background: linear-gradient(135deg, #fde68a, #fbbf24) !important;
            background-color: #fbbf24 !important;
            color: #0f172a !important;
            -webkit-text-fill-color: #0f172a !important;
            border-color: #b45309 !important;
            transform: translateY(-1px);
        }}

        [data-testid="stAppViewContainer"] .stButton > button:focus,
        [data-testid="stAppViewContainer"] .stDownloadButton > button:focus,
        [data-testid="stAppViewContainer"] div[data-testid="stFormSubmitButton"] button:focus,
        [data-testid="stAppViewContainer"] button[kind="primary"]:focus,
        [data-testid="stAppViewContainer"] button[kind="secondary"]:focus,
        [data-testid="stAppViewContainer"] button[data-testid="baseButton-secondary"]:focus,
        [data-testid="stAppViewContainer"] button[data-testid="baseButton-primary"]:focus {{
            outline: 2px solid rgba(217, 119, 6, 0.35) !important;
            outline-offset: 2px !important;
        }}

        [data-testid="stAppViewContainer"] .stButton > button:active,
        [data-testid="stAppViewContainer"] .stDownloadButton > button:active,
        [data-testid="stAppViewContainer"] div[data-testid="stFormSubmitButton"] button:active,
        [data-testid="stAppViewContainer"] button[kind="primary"]:active,
        [data-testid="stAppViewContainer"] button[kind="secondary"]:active,
        [data-testid="stAppViewContainer"] button[data-testid="baseButton-secondary"]:active,
        [data-testid="stAppViewContainer"] button[data-testid="baseButton-primary"]:active {{
            background: linear-gradient(135deg, #f59e0b, #ea580c) !important;
            background-color: #ea580c !important;
            color: #fff7ed !important;
            -webkit-text-fill-color: #fff7ed !important;
            border-color: #9a3412 !important;
            transform: translateY(0);
        }}

        [data-testid="stAppViewContainer"] .stButton > button > div,
        [data-testid="stAppViewContainer"] .stDownloadButton > button > div,
        [data-testid="stAppViewContainer"] div[data-testid="stFormSubmitButton"] button > div,
        [data-testid="stAppViewContainer"] button[kind="primary"] > div,
        [data-testid="stAppViewContainer"] button[kind="secondary"] > div,
        [data-testid="stAppViewContainer"] button[data-testid="baseButton-secondary"] > div,
        [data-testid="stAppViewContainer"] button[data-testid="baseButton-primary"] > div,
        [data-testid="stAppViewContainer"] .stButton > button > div > p,
        [data-testid="stAppViewContainer"] .stDownloadButton > button > div > p,
        [data-testid="stAppViewContainer"] div[data-testid="stFormSubmitButton"] button > div > p,
        [data-testid="stAppViewContainer"] button[kind="primary"] > div > p,
        [data-testid="stAppViewContainer"] button[kind="secondary"] > div > p,
        [data-testid="stAppViewContainer"] button[data-testid="baseButton-secondary"] > div > p,
        [data-testid="stAppViewContainer"] button[data-testid="baseButton-primary"] > div > p {{
            background: transparent !important;
            background-color: transparent !important;
        }}

        [data-testid="stAppViewContainer"] .stButton > button *,
        [data-testid="stAppViewContainer"] .stDownloadButton > button *,
        [data-testid="stAppViewContainer"] div[data-testid="stFormSubmitButton"] button *,
        [data-testid="stAppViewContainer"] button[kind="primary"] *,
        [data-testid="stAppViewContainer"] button[kind="secondary"] *,
        [data-testid="stAppViewContainer"] button[data-testid="baseButton-secondary"] *,
        [data-testid="stAppViewContainer"] button[data-testid="baseButton-primary"] * {{
            color: #10202a !important;
            -webkit-text-fill-color: #10202a !important;
            fill: #10202a !important;
            opacity: 1 !important;
            box-shadow: none !important;
        }}

        [data-testid="stAppViewContainer"] .stButton > button:active *,
        [data-testid="stAppViewContainer"] .stDownloadButton > button:active *,
        [data-testid="stAppViewContainer"] div[data-testid="stFormSubmitButton"] button:active *,
        [data-testid="stAppViewContainer"] button[kind="primary"]:active *,
        [data-testid="stAppViewContainer"] button[kind="secondary"]:active *,
        [data-testid="stAppViewContainer"] button[data-testid="baseButton-secondary"]:active *,
        [data-testid="stAppViewContainer"] button[data-testid="baseButton-primary"]:active * {{
            color: #fff7ed !important;
            -webkit-text-fill-color: #fff7ed !important;
            fill: #fff7ed !important;
        }}

        [data-testid="stAppViewContainer"] [role="radiogroup"] label,
        [data-testid="stAppViewContainer"] [role="radiogroup"] p {{
            color: var(--text) !important;
        }}

        [data-testid="stSidebar"] [role="radiogroup"] label,
        [data-testid="stSidebar"] [role="radiogroup"] p,
        [data-testid="stSidebar"] [data-baseweb="radio"] label {{
            color: #f8fafc !important;
        }}

        .hero {{
            padding: 1.8rem 2rem;
            border-radius: 28px;
            background:
                linear-gradient(135deg, rgba(15, 118, 110, 0.92), rgba(17, 24, 39, 0.9)),
                linear-gradient(120deg, rgba(245, 158, 11, 0.14), transparent 55%);
            color: #f8fafc;
            box-shadow: var(--shadow);
            overflow: hidden;
            position: relative;
        }}

        .hero::after {{
            content: "";
            position: absolute;
            inset: auto -6% -40% auto;
            width: 220px;
            height: 220px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(245, 158, 11, 0.35), transparent 70%);
        }}

        .hero h1 {{
            margin: 0;
            font-size: 2.4rem;
            line-height: 1.1;
        }}

        .hero p {{
            margin: 0.8rem 0 0;
            max-width: 760px;
            color: rgba(248, 250, 252, 0.86);
            font-size: 1rem;
        }}

        .section-card {{
            background: var(--panel);
            backdrop-filter: blur(10px);
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 1.2rem 1.3rem;
            box-shadow: var(--shadow);
        }}

        .metric-card {{
            background: var(--panel-strong);
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 1rem 1.1rem;
            box-shadow: var(--shadow);
            min-height: 132px;
        }}

        .metric-label {{
            color: var(--muted);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}

        .metric-value {{
            font-size: 2rem;
            font-weight: 700;
            margin-top: 0.35rem;
            color: var(--text);
        }}

        .metric-help {{
            color: var(--muted);
            font-size: 0.92rem;
            margin-top: 0.35rem;
        }}

        div[data-testid="stForm"] {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 1rem 1rem 0.2rem;
        }}

        div[data-testid="stForm"] label,
        div[data-testid="stForm"] p,
        div[data-testid="stForm"] span {{
            color: var(--text);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        """
        <div class="hero">
            <h1>PharmaEase Intelligence Hub</h1>
            <p>Explore pharmacy operations, forecast medicine demand, estimate staffing, predict sales, and recommend likely drugs from symptom text in one dashboard.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label, value, help_text):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_figure(fig):
    fig.update_layout(
        paper_bgcolor="rgba(255, 252, 247, 0.92)",
        plot_bgcolor="rgba(255, 252, 247, 0.78)",
        font=dict(color="#14232d"),
        title_font=dict(color="#14232d", size=22),
        legend=dict(
            bgcolor="rgba(255, 252, 247, 0.72)",
            font=dict(color="#14232d"),
        ),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_xaxes(
        title_font=dict(color="#425466"),
        tickfont=dict(color="#425466"),
        gridcolor="rgba(20, 35, 45, 0.12)",
        zerolinecolor="rgba(20, 35, 45, 0.12)",
    )
    fig.update_yaxes(
        title_font=dict(color="#425466"),
        tickfont=dict(color="#425466"),
        gridcolor="rgba(20, 35, 45, 0.12)",
        zerolinecolor="rgba(20, 35, 45, 0.12)",
    )
    return fig


def render_overview():
    sales_df = load_sales_data()
    employee_df = load_employee_data()
    prescription_df = load_prescription_data()
    inventory_df = load_inventory_data()

    total_revenue = sales_df["revenue"].sum()
    total_transactions = sales_df["transaction_id"].nunique()
    avg_staff = employee_df["staff_needed"].mean()
    tracked_drugs = inventory_df["drug_name"].nunique()

    st.subheader("Overview")
    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric_card("Total Revenue", f"Rs. {total_revenue:,.0f}", "Revenue from the sales dataset")
    with metric_cols[1]:
        render_metric_card("Transactions", f"{total_transactions:,}", "Recorded pharmacy transactions")
    with metric_cols[2]:
        render_metric_card("Average Staff", f"{avg_staff:.1f}", "Average staff requirement")
    with metric_cols[3]:
        render_metric_card("Tracked Drugs", f"{tracked_drugs}", "Medicines in inventory history")

    st.write("")
    chart_col, mix_col = st.columns((1.5, 1))

    with chart_col:
        daily_revenue = sales_df.groupby("date", as_index=False)["revenue"].sum()
        revenue_chart = px.area(
            daily_revenue,
            x="date",
            y="revenue",
            title="Daily Revenue Trend",
            color_discrete_sequence=["#0f766e"],
        )
        style_figure(revenue_chart)
        st.plotly_chart(revenue_chart, use_container_width=True)

    with mix_col:
        top_items = (
            sales_df.groupby("item", as_index=False)["quantity"]
            .sum()
            .sort_values("quantity", ascending=False)
            .head(6)
        )
        item_chart = px.pie(
            top_items,
            names="item",
            values="quantity",
            title="Top Medicines by Quantity",
            hole=0.55,
            color_discrete_sequence=px.colors.sequential.Tealgrn,
        )
        style_figure(item_chart)
        st.plotly_chart(item_chart, use_container_width=True)

    lower_left, lower_right = st.columns(2)

    with lower_left:
        staff_chart = px.scatter(
            employee_df,
            x="transactions",
            y="staff_needed",
            size="total_quantity",
            title="Staff Requirement vs Transactions",
            color="staff_needed",
            color_continuous_scale="Tealgrn",
        )
        style_figure(staff_chart)
        st.plotly_chart(staff_chart, use_container_width=True)

    with lower_right:
        common_drugs = (
            prescription_df["drug_prescribed"]
            .value_counts()
            .rename_axis("drug")
            .reset_index(name="count")
        )
        prescription_chart = px.bar(
            common_drugs.head(7),
            x="drug",
            y="count",
            title="Most Recommended Drugs",
            color="count",
            color_continuous_scale="Sunsetdark",
        )
        style_figure(prescription_chart)
        prescription_chart.update_layout(coloraxis_showscale=False)
        st.plotly_chart(prescription_chart, use_container_width=True)

    preview_col1, preview_col2 = st.columns(2)
    with preview_col1:
        st.markdown("### Sales Snapshot")
        st.dataframe(
            sales_df[["date", "item", "quantity", "price", "revenue"]].sort_values("date", ascending=False).head(8),
            use_container_width=True,
            hide_index=True,
        )
    with preview_col2:
        st.markdown("### Prescription Snapshot")
        st.dataframe(
            prescription_df.sort_values("date", ascending=False).head(8),
            use_container_width=True,
            hide_index=True,
        )


def render_sales_page():
    sales_df = load_sales_data()

    st.subheader("Sales Prediction")
    st.caption("Estimate expected sales revenue from calendar and transaction activity.")

    left_col, right_col = st.columns((1.1, 1))
    with left_col:
        with st.form("sales_prediction_form"):
            prediction_date = st.date_input("Prediction date")
            total_quantity = st.number_input("Total quantity sold", min_value=1, value=120, step=1)
            num_transactions = st.number_input("Number of transactions", min_value=1, value=30, step=1)
            submitted = st.form_submit_button("Predict sales", type="primary")

        if submitted:
            weekday = prediction_date.weekday()
            predicted_sales = predict_sales(
                day=prediction_date.day,
                month=prediction_date.month,
                weekday=weekday,
                quantity=int(total_quantity),
                transactions=int(num_transactions),
            )
            st.success(f"Predicted sales: Rs. {predicted_sales:,.2f}")

    with right_col:
        weekday_mix = (
            sales_df.groupby("day_name", as_index=False)["revenue"]
            .sum()
            .assign(
                day_name=lambda df: pd.Categorical(
                    df["day_name"], categories=WEEKDAY_OPTIONS, ordered=True
                )
            )
            .sort_values("day_name")
        )
        sales_chart = px.bar(
            weekday_mix,
            x="day_name",
            y="revenue",
            title="Revenue by Weekday",
            color="revenue",
            color_continuous_scale="Teal",
        )
        style_figure(sales_chart)
        sales_chart.update_layout(coloraxis_showscale=False)
        st.plotly_chart(sales_chart, use_container_width=True)

    monthly_item_sales = (
        sales_df.groupby(["month", "item"], as_index=False)["revenue"].sum().sort_values("month")
    )
    trend_chart = px.line(
        monthly_item_sales,
        x="month",
        y="revenue",
        color="item",
        title="Monthly Revenue by Item",
        markers=True,
    )
    style_figure(trend_chart)
    st.plotly_chart(trend_chart, use_container_width=True)


def render_staff_page():
    employee_df = load_employee_data()

    st.subheader("Staff Requirement")
    st.caption("Estimate the number of staff members needed for expected store workload.")

    left_col, right_col = st.columns((1.05, 1))
    with left_col:
        with st.form("staff_prediction_form"):
            weekday_name = st.selectbox("Day of week", WEEKDAY_OPTIONS)
            transactions = st.number_input("Transactions expected", min_value=1, value=45, step=1)
            total_quantity = st.number_input("Total quantity expected", min_value=1, value=150, step=1)
            submitted = st.form_submit_button("Estimate staff", type="primary")

        if submitted:
            predicted_staff = predict_staff(
                day=WEEKDAY_OPTIONS.index(weekday_name),
                transactions=int(transactions),
                quantity=int(total_quantity),
            )
            st.success(f"Recommended staff count: {predicted_staff}")

    with right_col:
        heatmap_data = (
            employee_df.groupby("day_of_week", as_index=False)["staff_needed"]
            .mean()
            .assign(day_name=lambda df: df["day_of_week"].map(lambda x: WEEKDAY_OPTIONS[int(x)]))
        )
        staff_day_chart = px.bar(
            heatmap_data,
            x="day_name",
            y="staff_needed",
            title="Average Staff Need by Weekday",
            color="staff_needed",
            color_continuous_scale="Agsunset",
        )
        style_figure(staff_day_chart)
        staff_day_chart.update_layout(coloraxis_showscale=False)
        st.plotly_chart(staff_day_chart, use_container_width=True)

    st.dataframe(employee_df.head(12), use_container_width=True, hide_index=True)


def render_prescription_page():
    prescription_df = load_prescription_data()

    st.subheader("Prescription Recommendation")
    st.caption("Enter symptoms and let the trained text model suggest the likely medicine.")

    left_col, right_col = st.columns((1.05, 1))
    with left_col:
        with st.form("prescription_form"):
            symptoms = st.text_area(
                "Symptoms",
                value="fever cough sore throat",
                height=140,
                placeholder="Type symptoms like fever, body pain, cough...",
            )
            submitted = st.form_submit_button("Recommend drug", type="primary")

        if submitted:
            if symptoms.strip():
                suggested_drug = predict_drug(symptoms.strip())
                st.success(f"Suggested drug: {suggested_drug}")
            else:
                st.warning("Please enter symptoms before requesting a recommendation.")

    with right_col:
        prescription_counts = (
            prescription_df["drug_prescribed"]
            .value_counts()
            .rename_axis("drug")
            .reset_index(name="count")
        )
        prescription_chart = px.funnel_area(
            prescription_counts.head(8),
            names="drug",
            values="count",
            title="Prescription Distribution",
            color_discrete_sequence=px.colors.sequential.Brwnyl,
        )
        style_figure(prescription_chart)
        st.plotly_chart(prescription_chart, use_container_width=True)

    st.markdown("### Recent Prescription Records")
    st.dataframe(
        prescription_df.sort_values("date", ascending=False).head(12),
        use_container_width=True,
        hide_index=True,
    )


def render_inventory_page():
    inventory_df = load_inventory_data()
    available_drugs = get_available_drugs()

    st.subheader("Inventory Forecast")
    st.caption("Forecast medicine demand using the pre-trained ARIMA model for each supported drug.")

    left_col, right_col = st.columns((1.05, 1))
    with left_col:
        with st.form("inventory_form"):
            drug_name = st.selectbox("Select drug", available_drugs)
            forecast_days = st.slider("Forecast horizon (days)", min_value=1, max_value=30, value=7)
            submitted = st.form_submit_button("Forecast inventory", type="primary")

        if submitted:
            forecast = predict_inventory(drug_name, forecast_days)
            forecast_df = pd.DataFrame(
                {
                    "day": range(1, forecast_days + 1),
                    "forecast_sales": [round(float(value), 2) for value in forecast],
                }
            )
            st.success(f"Forecast generated for {drug_name}")
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)

            forecast_chart = px.line(
                forecast_df,
                x="day",
                y="forecast_sales",
                title=f"Forecasted Demand for {drug_name}",
                markers=True,
            )
            forecast_chart.update_traces(line_color="#0f766e")
            style_figure(forecast_chart)
            st.plotly_chart(forecast_chart, use_container_width=True)

    with right_col:
        history = inventory_df.groupby("drug_name", as_index=False)["sales"].sum().sort_values(
            "sales", ascending=False
        )
        inventory_chart = px.bar(
            history,
            x="drug_name",
            y="sales",
            title="Historical Sales by Drug",
            color="sales",
            color_continuous_scale="Temps",
        )
        style_figure(inventory_chart)
        inventory_chart.update_layout(coloraxis_showscale=False)
        st.plotly_chart(inventory_chart, use_container_width=True)

    st.markdown("### Inventory History")
    st.dataframe(
        inventory_df.sort_values("date", ascending=False).head(12),
        use_container_width=True,
        hide_index=True,
    )


def main():
    inject_styles()
    render_hero()

    st.sidebar.title("PharmaEase")
    # st.sidebar.caption("ML dashboard for pharmacy operations")
    page = st.sidebar.radio(
        "Navigate", 
        ["Overview", "Sales", "Staff", "Prescription", "Inventory"],
    )
    # st.sidebar.markdown("---")
    # st.sidebar.info(
    #     "Models are loaded from the local `models/` folder and the charts use CSV files from `data/raw/`."
    # )

    if page == "Overview":
        render_overview()
    elif page == "Sales":
        render_sales_page()
    elif page == "Staff":
        render_staff_page()
    elif page == "Prescription":
        render_prescription_page()
    else:
        render_inventory_page()


if __name__ == "__main__":
    main()
