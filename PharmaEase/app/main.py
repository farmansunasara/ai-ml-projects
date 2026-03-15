"""
PharmaEase — Pharmacy Management System
Phase 4: Streamlit Dashboard
Company: Brainybeam Info-Tech PVT LTD
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime, date
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────
st.set_page_config(
    page_title="PharmaEase",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────
st.markdown("""
<style>
    /* ════════════════════════════════════════
       SIDEBAR — always dark navy
    ════════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a2f4b 0%, #0f1e30 100%) !important;
    }
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stCaption {
        color: #c8d8e8 !important;
    }
    [data-testid="stSidebar"] .stMarkdown strong {
        color: #ffffff !important;
    }

    /* ════════════════════════════════════════
       METRIC CARDS
       Use currentColor so text inherits from
       Streamlit's own light/dark token.
    ════════════════════════════════════════ */
    .metric-card {
        border-radius: 12px;
        padding: 18px 22px;
        border-left: 4px solid #2563eb;
        margin-bottom: 6px;
        /* Streamlit's surface token — works
           on both light (#ffffff) and dark (#1e1e2e) */
        background-color: var(--secondary-background-color);
    }
    .metric-card-green  { border-left-color: #22c55e !important; }
    .metric-card-red    { border-left-color: #ef4444 !important; }
    .metric-card-amber  { border-left-color: #f59e0b !important; }
    .metric-card-purple { border-left-color: #a855f7 !important; }

    .metric-val {
        font-size: 26px;
        font-weight: 700;
        /* currentColor = whatever Streamlit sets for text */
        color: currentColor;
        margin: 6px 0 2px;
        line-height: 1.1;
    }
    .metric-lbl {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.55;
    }
    .metric-sub {
        font-size: 11px;
        opacity: 0.45;
        margin-top: 3px;
    }

    /* ════════════════════════════════════════
       SECTION HEADER BANNER
    ════════════════════════════════════════ */
    .section-header {
        border-radius: 12px;
        padding: 16px 24px;
        margin-bottom: 20px;
        border-top: 4px solid #2563eb;
        background-color: var(--secondary-background-color);
    }
    .section-header h2 {
        margin: 0 0 4px;
        font-size: 21px;
        font-weight: 700;
        color: currentColor;
    }
    .section-header p {
        margin: 0;
        font-size: 13px;
        opacity: 0.55;
    }

    /* ════════════════════════════════════════
       BADGES  — semi-transparent so they read
       on both light & dark backgrounds
    ════════════════════════════════════════ */
    .badge-red {
        display: inline-block;
        background: rgba(239,68,68,0.15);
        color: #ef4444;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        border: 1px solid rgba(239,68,68,0.35);
    }
    .badge-green {
        display: inline-block;
        background: rgba(34,197,94,0.15);
        color: #22c55e;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        border: 1px solid rgba(34,197,94,0.35);
    }
    .badge-amber {
        display: inline-block;
        background: rgba(245,158,11,0.15);
        color: #f59e0b;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        border: 1px solid rgba(245,158,11,0.35);
    }

    /* ════════════════════════════════════════
       PREDICTION BOX — always dark blue,
       always white text inside
    ════════════════════════════════════════ */
    .pred-box {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
        border-radius: 14px;
        padding: 22px 18px;
        text-align: center;
        margin: 14px 0;
    }
    .pred-box .pred-val {
        font-size: 38px;
        font-weight: 800;
        color: #ffffff !important;
        display: block;
        line-height: 1.1;
    }
    .pred-box .pred-lbl {
        font-size: 12px;
        color: rgba(255,255,255,0.80) !important;
        display: block;
        margin-top: 6px;
    }

    /* ════════════════════════════════════════
       PLOTLY CHARTS — transparent so they
       inherit the page/card background
    ════════════════════════════════════════ */
    .js-plotly-plot .plotly,
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }

    /* ════════════════════════════════════════
       HIDE STREAMLIT CHROME
    ════════════════════════════════════════ */
    #MainMenu       { visibility: hidden; }
    footer          { visibility: hidden; }
    .stDeployButton { display: none;      }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# DATA LOADERS (cached)
# ══════════════════════════════════════════════

@st.cache_data
def load_data():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base, 'data')
    return {
        'medicines':     pd.read_csv(f'{data_dir}/medicines.csv'),
        'sales':         pd.read_csv(f'{data_dir}/sales.csv',
                                     parse_dates=['date']),
        'prescriptions': pd.read_csv(f'{data_dir}/prescriptions.csv',
                                     parse_dates=['date']),
        'employees':     pd.read_csv(f'{data_dir}/employees.csv',
                                     parse_dates=['join_date']),
        'customers':     pd.read_csv(f'{data_dir}/customers.csv'),
    }

@st.cache_resource
def load_models():
    base      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base, 'models')
    models = {}
    files = {
        'model1':    'model1_demand_forecast.pkl',
        'model2':    'model2_stockout_prediction.pkl',
        'model3':    'model3_sales_prediction.pkl',
        'enc_cat':   'encoder_category.pkl',
        'enc_med':   'encoder_medicine.pkl',
        'enc_sea':   'encoder_season.pkl',
        'enc_cat2':  'encoder_category_m2.pkl',
    }
    for key, fname in files.items():
        path = os.path.join(model_dir, fname)
        if os.path.exists(path):
            models[key] = joblib.load(path)
    return models

data   = load_data()
models = load_models()

df_med  = data['medicines'].copy()
df_sal  = data['sales'].copy()
df_prx  = data['prescriptions'].copy()
df_emp  = data['employees'].copy()
df_cust = data['customers'].copy()

# Pre-compute stock fields
df_sal['year']   = df_sal['date'].dt.year
df_sal['month']  = df_sal['date'].dt.month
df_sal['season'] = df_sal['month'].map({
    1:'Winter',2:'Winter',3:'Summer',4:'Summer',5:'Summer',
    6:'Monsoon',7:'Monsoon',8:'Monsoon',9:'Monsoon',
    10:'Autumn',11:'Winter',12:'Winter'
})

today = pd.Timestamp(date.today())
df_med['expiry_date']    = pd.to_datetime(df_med['expiry_date'])
df_med['days_to_expiry'] = (df_med['expiry_date'] - today).dt.days
date_range = (df_sal['date'].max() - df_sal['date'].min()).days + 1
avg_daily  = (df_sal.groupby('medicine_id')['quantity']
              .sum().div(date_range).reset_index())
avg_daily.columns = ['medicine_id', 'avg_daily_sales']
df_med = df_med.merge(avg_daily, on='medicine_id', how='left')
df_med['avg_daily_sales'] = df_med['avg_daily_sales'].fillna(0.01)
df_med['days_remaining']  = (df_med['stock_qty'] / df_med['avg_daily_sales']).round(0)
df_med['stock_status'] = df_med.apply(
    lambda r: 'Critical' if r['stock_qty'] <= r['reorder_level']
    else ('Low' if r['stock_qty'] <= r['reorder_level'] * 1.5 else 'OK'), axis=1)


# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 💊 PharmaEase")
    st.markdown("*Pharmacy Management System*")
    st.markdown("---")

    page = st.radio("Navigation", [
        "🏠  Dashboard",
        "📦  Inventory Management",
        "📋  Prescription Tracking",
        "💰  Sales & Billing",
        "👥  Employee Management",
        "📊  Reports & Analytics",
    ])

    st.markdown("---")
    st.markdown("**ML Models Active**")
    for m, label in [
        ('model1', '📈 Demand Forecast'),
        ('model2', '🔴 Stock-out Predict'),
        ('model3', '💵 Revenue Predict'),
    ]:
        status = "🟢 Loaded" if m in models else "🔴 Not found"
        st.markdown(f"`{label}` {status}")

    st.markdown("---")
    st.caption("Brainybeam Info-Tech PVT LTD")
    st.caption("Data Science Internship — 2026")


# ══════════════════════════════════════════════
# PAGE 1 — DASHBOARD (HOME)
# ══════════════════════════════════════════════

if "Dashboard" in page:
    st.markdown("""
    <div class='section-header'>
        <h2>🏠 PharmaEase Dashboard</h2>
        <p>Real-time overview of pharmacy operations — powered by Data Science & ML</p>
    </div>""", unsafe_allow_html=True)

    # ── KPI Row ───────────────────────────────
    total_rev   = df_sal['total_price'].sum()
    total_trans = len(df_sal)
    critical    = len(df_med[df_med['stock_status'] == 'Critical'])
    expiring    = len(df_med[df_med['days_to_expiry'] <= 90])
    total_rx    = len(df_prx)
    total_staff = len(df_emp)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-lbl'>Total Revenue</div>
            <div class='metric-val'>₹{total_rev/100000:.1f}L</div>
            <div class='metric-sub'>2023–2024</div></div>""",
            unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card metric-card-green'>
            <div class='metric-lbl'>Transactions</div>
            <div class='metric-val'>{total_trans:,}</div>
            <div class='metric-sub'>All time</div></div>""",
            unsafe_allow_html=True)
    with c3:
        color = 'red' if critical > 0 else 'green'
        st.markdown(f"""<div class='metric-card metric-card-{color}'>
            <div class='metric-lbl'>Critical Stock</div>
            <div class='metric-val'>{critical}</div>
            <div class='metric-sub'>Need reorder now</div></div>""",
            unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='metric-card metric-card-amber'>
            <div class='metric-lbl'>Expiring Soon</div>
            <div class='metric-val'>{expiring}</div>
            <div class='metric-sub'>Within 90 days</div></div>""",
            unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class='metric-card metric-card-purple'>
            <div class='metric-lbl'>Prescriptions</div>
            <div class='metric-val'>{total_rx:,}</div>
            <div class='metric-sub'>Total records</div></div>""",
            unsafe_allow_html=True)
    with c6:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-lbl'>Total Staff</div>
            <div class='metric-val'>{total_staff:,}</div>
            <div class='metric-sub'>All roles</div></div>""",
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts Row ────────────────────────────
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📈 Monthly Revenue Trend")
        monthly = (df_sal.groupby(df_sal['date'].dt.to_period('M'))['total_price']
                   .sum().reset_index())
        monthly['date'] = monthly['date'].astype(str)
        fig = px.area(monthly, x='date', y='total_price',
                      color_discrete_sequence=['#2563eb'],
                      labels={'total_price': 'Revenue (₹)', 'date': 'Month'})
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis_tickprefix='₹', height=280)
        fig.update_xaxes(tickangle=45, tickfont_size=9, tickfont_color=None)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🏷️ Category Revenue Share")
        cat_rev = (df_sal.groupby('category')['total_price']
                   .sum().sort_values(ascending=False))
        top7    = cat_rev.head(7)
        other   = cat_rev.iloc[7:].sum()
        pie_data = pd.concat([top7, pd.Series({'Others': other})])
        fig2 = px.pie(values=pie_data.values, names=pie_data.index,
                      color_discrete_sequence=px.colors.qualitative.Set2,
                      hole=0.4)
        fig2.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)', height=280,
            legend=dict(font_size=10))
        fig2.update_traces(textfont_size=10)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Alerts + YoY ─────────────────────────
    col3, col4 = st.columns([1, 2])

    with col3:
        st.subheader("🚨 Active Alerts")
        alerts = df_med[df_med['stock_status'].isin(['Critical','Low'])][
            ['name','stock_status','stock_qty','reorder_level']].head(8)
        if len(alerts) == 0:
            st.success("All stock levels are healthy!")
        for _, row in alerts.iterrows():
            badge = 'badge-red' if row['stock_status'] == 'Critical' else 'badge-amber'
            st.markdown(
                f"<span class='{badge}'>{row['stock_status']}</span> "
                f"**{row['name']}** — {row['stock_qty']} units",
                unsafe_allow_html=True)

        st.markdown("---")
        exp_alert = df_med[df_med['days_to_expiry'] <= 90].sort_values('days_to_expiry')
        if len(exp_alert) > 0:
            st.markdown("**⏰ Expiring <90 days:**")
            for _, row in exp_alert.head(5).iterrows():
                st.markdown(
                    f"<span class='badge-red'>{row['days_to_expiry']}d</span> "
                    f"{row['name']}",
                    unsafe_allow_html=True)

    with col4:
        st.subheader("📊 Year-over-Year Comparison")
        yoy = (df_sal.groupby(['year','month'])['total_price']
               .sum().reset_index())
        month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                       7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
        yoy['month_name'] = yoy['month'].map(month_names)
        fig3 = px.line(yoy, x='month_name', y='total_price',
                       color='year', markers=True,
                       color_discrete_sequence=['#2563eb','#dc2626'],
                       labels={'total_price':'Revenue (₹)','month_name':'Month',
                               'year':'Year'})
        fig3.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis_tickprefix='₹', height=280,
            xaxis={'categoryorder':'array',
                   'categoryarray':list(month_names.values())})
        st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE 2 — INVENTORY MANAGEMENT
# ══════════════════════════════════════════════

elif "Inventory" in page:
    st.markdown("""
    <div class='section-header'>
        <h2>📦 Inventory Management</h2>
        <p>Real-time stock levels · Expiry tracking · ML-powered stock-out prediction</p>
    </div>""", unsafe_allow_html=True)

    # ── Filters ───────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        cat_filter = st.selectbox("Filter by Category",
            ['All'] + sorted(df_med['category'].unique()))
    with col2:
        status_filter = st.selectbox("Stock Status",
            ['All', 'Critical', 'Low', 'OK'])
    with col3:
        supplier_filter = st.selectbox("Supplier",
            ['All'] + sorted(df_med['supplier'].unique()))

    df_inv = df_med.copy()
    if cat_filter    != 'All': df_inv = df_inv[df_inv['category']  == cat_filter]
    if status_filter != 'All': df_inv = df_inv[df_inv['stock_status'] == status_filter]
    if supplier_filter != 'All': df_inv = df_inv[df_inv['supplier'] == supplier_filter]

    # ── KPIs ──────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-lbl'>Total Medicines</div>
            <div class='metric-val'>{len(df_inv)}</div></div>""",
            unsafe_allow_html=True)
    with c2:
        crit_n = len(df_inv[df_inv['stock_status']=='Critical'])
        st.markdown(f"""<div class='metric-card metric-card-red'>
            <div class='metric-lbl'>Critical Stock</div>
            <div class='metric-val'>{crit_n}</div></div>""",
            unsafe_allow_html=True)
    with c3:
        exp_n = len(df_inv[df_inv['days_to_expiry'] <= 90])
        st.markdown(f"""<div class='metric-card metric-card-amber'>
            <div class='metric-lbl'>Expiring <90 days</div>
            <div class='metric-val'>{exp_n}</div></div>""",
            unsafe_allow_html=True)
    with c4:
        total_units = df_inv['stock_qty'].sum()
        st.markdown(f"""<div class='metric-card metric-card-green'>
            <div class='metric-lbl'>Total Units</div>
            <div class='metric-val'>{total_units:,}</div></div>""",
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Stock Level Chart ─────────────────────
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("📊 Stock Levels vs Reorder Level")
        color_map = {'Critical':'#dc2626','Low':'#d97706','OK':'#16a34a'}
        fig = px.bar(df_inv.sort_values('stock_qty'),
                     x='stock_qty', y='name',
                     color='stock_status',
                     color_discrete_map=color_map,
                     orientation='h',
                     labels={'stock_qty':'Units in Stock','name':'Medicine'})
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            height=max(300, len(df_inv)*18),
            margin=dict(l=10, r=10, t=10, b=10),
            legend_title='Status', yaxis_title='')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🔮 ML Stock-out Prediction")
        st.markdown("*Predict if a medicine will stock out within 90 days*")

        if 'model2' in models and 'enc_cat2' in models:
            sel_med = st.selectbox("Select Medicine",
                                   df_med['name'].tolist())
            row = df_med[df_med['name'] == sel_med].iloc[0]

            try:
                cat_enc = models['enc_cat2'].transform([row['category']])[0]
                features = np.array([[
                    row['stock_qty'],
                    row['avg_daily_sales'],
                    row['days_remaining'],
                    row['reorder_level'],
                    cat_enc,
                    row['days_to_expiry'],
                    row['price']
                ]])
                pred  = models['model2'].predict(features)[0]
                prob  = models['model2'].predict_proba(features)[0]

                if pred == 1:
                    st.error(f"🔴 **STOCK-OUT RISK**\n\nConfidence: {prob[1]*100:.0f}%")
                    st.markdown(f"Days remaining: **{row['days_remaining']:.0f}**")
                    st.markdown(f"Reorder from: **{row['supplier']}**")
                else:
                    st.success(f"🟢 **STOCK SAFE**\n\nConfidence: {prob[0]*100:.0f}%")
                    st.markdown(f"Days remaining: **{row['days_remaining']:.0f}**")
            except Exception as e:
                st.warning(f"Prediction error: {e}")
        else:
            st.warning("Model not loaded. Run Phase 3 first.")

    # ── Inventory Table ───────────────────────
    st.subheader("📋 Inventory Table")
    display_cols = ['name','category','price','stock_qty','reorder_level',
                    'stock_status','avg_daily_sales','days_remaining',
                    'days_to_expiry','supplier']
    display_df = df_inv[display_cols].copy()
    display_df['avg_daily_sales'] = display_df['avg_daily_sales'].round(2)
    display_df['days_remaining']  = display_df['days_remaining'].astype(int)

    def color_status(val):
        if val == 'Critical':
            return 'color:#ef4444; font-weight:700'
        elif val == 'Low':
            return 'color:#f59e0b; font-weight:700'
        return 'color:#22c55e; font-weight:700'

    styled = display_df.style.applymap(color_status, subset=['stock_status'])
    st.dataframe(styled, use_container_width=True, height=400)


# ══════════════════════════════════════════════
# PAGE 3 — PRESCRIPTION TRACKING
# ══════════════════════════════════════════════

elif "Prescription" in page:
    st.markdown("""
    <div class='section-header'>
        <h2>📋 Prescription Tracking</h2>
        <p>Patient prescription records · Doctor analysis · Medicine patterns</p>
    </div>""", unsafe_allow_html=True)

    # ── Filters ───────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        doc_filter = st.selectbox("Filter by Doctor",
            ['All'] + sorted(df_prx['doctor'].unique()))
    with col2:
        city_filter = st.selectbox("Filter by City",
            ['All'] + sorted(df_prx['city'].unique()))
    with col3:
        gender_filter = st.selectbox("Gender",
            ['All', 'Male', 'Female'])

    df_rx = df_prx.copy()
    if doc_filter    != 'All': df_rx = df_rx[df_rx['doctor'] == doc_filter]
    if city_filter   != 'All': df_rx = df_rx[df_rx['city']   == city_filter]
    if gender_filter != 'All': df_rx = df_rx[df_rx['gender'] == gender_filter]

    # ── KPIs ──────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-lbl'>Total Prescriptions</div>
            <div class='metric-val'>{len(df_rx):,}</div></div>""",
            unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card metric-card-purple'>
            <div class='metric-lbl'>Unique Patients</div>
            <div class='metric-val'>{df_rx['patient_name'].nunique():,}</div></div>""",
            unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card metric-card-green'>
            <div class='metric-lbl'>Doctors</div>
            <div class='metric-val'>{df_rx['doctor'].nunique()}</div></div>""",
            unsafe_allow_html=True)
    with c4:
        avg_age = df_rx['age'].mean()
        st.markdown(f"""<div class='metric-card metric-card-amber'>
            <div class='metric-lbl'>Avg Patient Age</div>
            <div class='metric-val'>{avg_age:.0f} yrs</div></div>""",
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("👨‍⚕️ Prescriptions by Doctor")
        doc_counts = df_rx['doctor'].value_counts().head(10).reset_index()
        doc_counts.columns = ['Doctor','Prescriptions']
        fig = px.bar(doc_counts, x='Prescriptions', y='Doctor',
                     orientation='h',
                     color='Prescriptions',
                     color_continuous_scale='Blues')
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          height=320, margin=dict(l=10,r=10,t=10,b=10),
                          coloraxis_showscale=False, yaxis_title='')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("💊 Top Prescribed Medicines")
        med_counts = df_rx['medicine_name'].value_counts().head(10).reset_index()
        med_counts.columns = ['Medicine','Count']
        fig2 = px.bar(med_counts, x='Count', y='Medicine',
                      orientation='h',
                      color='Count',
                      color_continuous_scale='Greens')
        fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                           height=320, margin=dict(l=10,r=10,t=10,b=10),
                           coloraxis_showscale=False, yaxis_title='')
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("👥 Patient Age Distribution")
        df_rx['age_group'] = pd.cut(df_rx['age'],
            bins=[0,18,30,45,60,100],
            labels=['<18','18-30','31-45','46-60','60+'])
        age_dist = df_rx['age_group'].value_counts().sort_index().reset_index()
        age_dist.columns = ['Age Group','Count']
        fig3 = px.bar(age_dist, x='Age Group', y='Count',
                      color='Count', color_continuous_scale='Purples')
        fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                           height=280, margin=dict(l=10,r=10,t=10,b=10),
                           coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("📍 Prescriptions by City")
        city_dist = df_rx['city'].value_counts().reset_index()
        city_dist.columns = ['City','Count']
        fig4 = px.pie(city_dist, values='Count', names='City',
                      color_discrete_sequence=px.colors.qualitative.Pastel,
                      hole=0.4)
        fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=280,
                           margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig4, use_container_width=True)

    # ── Search & Table ────────────────────────
    st.subheader("🔍 Search Prescriptions")
    search = st.text_input("Search by patient name or medicine")
    df_show = df_rx.copy()
    if search:
        mask = (df_show['patient_name'].str.contains(search, case=False, na=False) |
                df_show['medicine_name'].str.contains(search, case=False, na=False))
        df_show = df_show[mask]

    st.dataframe(df_show[['prescription_id','patient_name','age','gender',
                           'doctor','date','medicine_name','dosage',
                           'duration','city']].reset_index(drop=True),
                 use_container_width=True, height=350)


# ══════════════════════════════════════════════
# PAGE 4 — SALES & BILLING
# ══════════════════════════════════════════════

elif "Sales" in page:
    st.markdown("""
    <div class='section-header'>
        <h2>💰 Sales & Billing</h2>
        <p>Transaction analysis · Revenue trends · ML revenue prediction</p>
    </div>""", unsafe_allow_html=True)

    # ── Filters ───────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        year_filter = st.selectbox("Year",
            ['All'] + [str(y) for y in sorted(df_sal['year'].unique())])
    with col2:
        cat_filter  = st.selectbox("Category",
            ['All'] + sorted(df_sal['category'].unique()))
    with col3:
        q_filter    = st.selectbox("Quarter",
            ['All','Q1','Q2','Q3','Q4'])

    df_s = df_sal.copy()
    if year_filter != 'All': df_s = df_s[df_s['year'] == int(year_filter)]
    if cat_filter  != 'All': df_s = df_s[df_s['category'] == cat_filter]
    if q_filter    != 'All':
        qmap = {'Q1':1,'Q2':2,'Q3':3,'Q4':4}
        df_s = df_s[df_s['quarter'] == qmap[q_filter]]

    # ── KPIs ──────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-lbl'>Total Revenue</div>
            <div class='metric-val'>₹{df_s['total_price'].sum()/1000:.1f}K</div></div>""",
            unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card metric-card-green'>
            <div class='metric-lbl'>Transactions</div>
            <div class='metric-val'>{len(df_s):,}</div></div>""",
            unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card metric-card-purple'>
            <div class='metric-lbl'>Avg Sale Value</div>
            <div class='metric-val'>₹{df_s['total_price'].mean():.0f}</div></div>""",
            unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='metric-card metric-card-amber'>
            <div class='metric-lbl'>Units Sold</div>
            <div class='metric-val'>{df_s['quantity'].sum():,}</div></div>""",
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ────────────────────────────────
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📈 Revenue Over Time")
        rev_time = (df_s.groupby(df_s['date'].dt.to_period('M'))['total_price']
                    .sum().reset_index())
        rev_time['date'] = rev_time['date'].astype(str)
        fig = px.bar(rev_time, x='date', y='total_price',
                     color_discrete_sequence=['#2563eb'],
                     labels={'total_price':'Revenue (₹)','date':'Month'})
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          height=280, margin=dict(l=10,r=10,t=10,b=10),
                          yaxis_tickprefix='₹')
        fig.update_xaxes(tickangle=45, tickfont_size=8, tickfont_color=None)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🔮 Revenue Predictor")
        st.markdown("*ML Model 3 — predict sale revenue*")
        if 'model3' in models and 'enc_cat' in models and 'enc_sea' in models:
            p_month = st.slider("Month", 1, 12, 6)
            p_cat   = st.selectbox("Category",
                sorted(df_sal['category'].unique()), key='pred_cat')
            p_qty   = st.slider("Quantity", 1, 7, 2)
            p_disc  = st.selectbox("Discount %", [0, 5, 10, 15])

            season_map = {1:'Winter',2:'Winter',3:'Summer',4:'Summer',
                          5:'Summer',6:'Monsoon',7:'Monsoon',8:'Monsoon',
                          9:'Monsoon',10:'Autumn',11:'Winter',12:'Winter'}
            p_season  = season_map[p_month]
            p_quarter = (p_month - 1) // 3 + 1
            avg_price = df_sal[df_sal['category']==p_cat]['unit_price'].mean()

            try:
                cat_enc = models['enc_cat'].transform([p_cat])[0]
                sea_enc = models['enc_sea'].transform([p_season])[0]
                X_pred  = np.array([[p_month, p_quarter, sea_enc,
                                     cat_enc, avg_price, p_disc, p_qty]])
                pred_rev = models['model3'].predict(X_pred)[0]
                st.markdown(f"""<div class='pred-box'>
                    <div class='pred-lbl'>Predicted Revenue</div>
                    <div class='pred-val'>₹{pred_rev:,.0f}</div>
                    <div class='pred-lbl'>{p_cat} · {p_season} · Qty {p_qty}</div>
                </div>""", unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Prediction error: {e}")
        else:
            st.warning("Model not loaded.")

    # ── Top medicines + Category ───────────────
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("🏆 Top 10 Medicines by Revenue")
        top_med = (df_s.groupby('medicine_name')['total_price']
                   .sum().sort_values(ascending=False).head(10).reset_index())
        fig2 = px.bar(top_med, x='total_price', y='medicine_name',
                      orientation='h',
                      color='total_price',
                      color_continuous_scale='Blues')
        fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                           height=320, margin=dict(l=10,r=10,t=10,b=10),
                           coloraxis_showscale=False,
                           yaxis_title='', xaxis_tickprefix='₹')
        st.plotly_chart(fig2, use_container_width=True)

    with col4:
        st.subheader("📦 Revenue by Category")
        cat_rev = (df_s.groupby('category')['total_price']
                   .sum().sort_values(ascending=False).reset_index())
        fig3 = px.bar(cat_rev, x='total_price', y='category',
                      orientation='h',
                      color='total_price',
                      color_continuous_scale='Greens')
        fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                           height=320, margin=dict(l=10,r=10,t=10,b=10),
                           coloraxis_showscale=False,
                           yaxis_title='', xaxis_tickprefix='₹')
        st.plotly_chart(fig3, use_container_width=True)

    # ── Transactions Table ────────────────────
    st.subheader("🧾 Recent Transactions")
    st.dataframe(
        df_s[['sale_id','date','medicine_name','category',
              'quantity','unit_price','discount_pct',
              'total_price','employee_id']
        ].sort_values('date', ascending=False).head(100).reset_index(drop=True),
        use_container_width=True, height=350)


# ══════════════════════════════════════════════
# PAGE 5 — EMPLOYEE MANAGEMENT
# ══════════════════════════════════════════════

elif "Employee" in page:
    st.markdown("""
    <div class='section-header'>
        <h2>👥 Employee Management</h2>
        <p>Staff profiles · Performance tracking · Shift analysis</p>
    </div>""", unsafe_allow_html=True)

    # ── Filters ───────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        role_filter  = st.selectbox("Role",
            ['All'] + sorted(df_emp['role'].unique()))
    with col2:
        shift_filter = st.selectbox("Shift",
            ['All'] + sorted(df_emp['shift'].unique()))
    with col3:
        city_e_filter = st.selectbox("City",
            ['All'] + sorted(df_emp['city'].unique()))

    df_e = df_emp.copy()
    if role_filter   != 'All': df_e = df_e[df_e['role']  == role_filter]
    if shift_filter  != 'All': df_e = df_e[df_e['shift'] == shift_filter]
    if city_e_filter != 'All': df_e = df_e[df_e['city']  == city_e_filter]

    # ── KPIs ──────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-lbl'>Total Employees</div>
            <div class='metric-val'>{len(df_e):,}</div></div>""",
            unsafe_allow_html=True)
    with c2:
        avg_sal = df_e['salary'].mean()
        st.markdown(f"""<div class='metric-card metric-card-green'>
            <div class='metric-lbl'>Avg Salary</div>
            <div class='metric-val'>₹{avg_sal/1000:.1f}K</div></div>""",
            unsafe_allow_html=True)
    with c3:
        avg_perf = df_e['performance_score'].mean()
        st.markdown(f"""<div class='metric-card metric-card-purple'>
            <div class='metric-lbl'>Avg Performance</div>
            <div class='metric-val'>{avg_perf:.1f}</div>
            <div class='metric-sub'>out of 100</div></div>""",
            unsafe_allow_html=True)
    with c4:
        top_perf = df_e['performance_score'].max()
        st.markdown(f"""<div class='metric-card metric-card-amber'>
            <div class='metric-lbl'>Top Score</div>
            <div class='metric-val'>{top_perf:.1f}</div></div>""",
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("💼 Role Distribution")
        role_dist = df_e['role'].value_counts().reset_index()
        role_dist.columns = ['Role','Count']
        fig = px.pie(role_dist, values='Count', names='Role',
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     hole=0.4)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=280,
                          margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("💰 Avg Salary by Role")
        sal_role = (df_e.groupby('role')['salary']
                    .mean().sort_values().reset_index())
        fig2 = px.bar(sal_role, x='salary', y='role',
                      orientation='h',
                      color='salary',
                      color_continuous_scale='Purples')
        fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                           height=280, margin=dict(l=10,r=10,t=10,b=10),
                           coloraxis_showscale=False,
                           yaxis_title='', xaxis_tickprefix='₹')
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("⭐ Performance by Shift")
        shift_perf = (df_e.groupby('shift')['performance_score']
                      .mean().reset_index())
        shift_perf.columns = ['Shift','Avg Score']
        fig3 = px.bar(shift_perf, x='Shift', y='Avg Score',
                      color='Avg Score',
                      color_continuous_scale='Blues',
                      range_y=[70, 80])
        fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                           height=260, margin=dict(l=10,r=10,t=10,b=10),
                           coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("📊 Performance Distribution")
        fig4 = px.histogram(df_e, x='performance_score',
                            nbins=20, color_discrete_sequence=['#7c3aed'])
        fig4.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                           height=260, margin=dict(l=10,r=10,t=10,b=10),
                           xaxis_title='Performance Score',
                           yaxis_title='Count')
        st.plotly_chart(fig4, use_container_width=True)

    # ── Top/Bottom performers ─────────────────
    col5, col6 = st.columns(2)
    with col5:
        st.subheader("🏆 Top 10 Performers")
        top10 = df_e.nlargest(10,'performance_score')[
            ['name','role','shift','performance_score']].reset_index(drop=True)
        st.dataframe(top10, use_container_width=True)

    with col6:
        st.subheader("⚠️ Need Support (Bottom 10)")
        bot10 = df_e.nsmallest(10,'performance_score')[
            ['name','role','shift','performance_score']].reset_index(drop=True)
        st.dataframe(bot10, use_container_width=True)

    # ── Full Table ────────────────────────────
    st.subheader("👤 Employee Directory")
    search_e = st.text_input("Search by name or role")
    df_show  = df_e.copy()
    if search_e:
        mask = (df_show['name'].str.contains(search_e, case=False, na=False) |
                df_show['role'].str.contains(search_e, case=False, na=False))
        df_show = df_show[mask]
    st.dataframe(df_show.reset_index(drop=True),
                 use_container_width=True, height=350)


# ══════════════════════════════════════════════
# PAGE 6 — REPORTS & ANALYTICS
# ══════════════════════════════════════════════

elif "Reports" in page:
    st.markdown("""
    <div class='section-header'>
        <h2>📊 Reports & Analytics</h2>
        <p>Seasonal analysis · Demand forecasting · Business intelligence</p>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📅 Seasonal Analysis",
        "🔮 Demand Forecast",
        "🌍 Geographic Analysis",
        "📈 Business Summary"
    ])

    # ── TAB 1: Seasonal ───────────────────────
    with tab1:
        st.subheader("🌦️ Seasonal Demand Heatmap")
        heatmap = (df_sal.groupby(['category','month'])['quantity']
                   .sum().unstack(fill_value=0))
        heatmap.columns = ['Jan','Feb','Mar','Apr','May','Jun',
                           'Jul','Aug','Sep','Oct','Nov','Dec']
        fig, ax = plt.subplots(figsize=(14, 7))
        sns.heatmap(heatmap, annot=True, fmt='d', cmap='YlOrRd',
                    linewidths=0.5, linecolor='white', ax=ax,
                    cbar_kws={'label':'Units Sold'})
        ax.set_title('Medicine Category Demand by Month', fontsize=14,
                     fontweight='bold', pad=12)
        ax.tick_params(axis='y', rotation=0)
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("🍂 Season-wise Performance")
        season_map_s = {1:'Winter',2:'Winter',3:'Summer',4:'Summer',
                        5:'Summer',6:'Monsoon',7:'Monsoon',8:'Monsoon',
                        9:'Monsoon',10:'Autumn',11:'Winter',12:'Winter'}
        df_sal['season'] = df_sal['month'].map(season_map_s)
        season_stats = (df_sal.groupby('season')
                        .agg(revenue=('total_price','sum'),
                             transactions=('sale_id','count'),
                             units=('quantity','sum'))
                        .reset_index())
        season_order  = ['Winter','Summer','Monsoon','Autumn']
        season_stats['season'] = pd.Categorical(
            season_stats['season'], categories=season_order, ordered=True)
        season_stats = season_stats.sort_values('season')

        col1, col2, col3 = st.columns(3)
        for col, metric, label, prefix in [
            (col1, 'revenue',      'Revenue',       '₹'),
            (col2, 'transactions', 'Transactions',  ''),
            (col3, 'units',        'Units Sold',     '')
        ]:
            with col:
                fig_s = px.bar(season_stats, x='season', y=metric,
                               color='season',
                               color_discrete_sequence=['#5B9BD5','#FFC000',
                                                         '#70AD47','#FF6B6B'])
                fig_s.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    height=280, margin=dict(l=10,r=10,t=30,b=10),
                    title=label, showlegend=False,
                    yaxis_tickprefix=prefix)
                st.plotly_chart(fig_s, use_container_width=True)

    # ── TAB 2: Demand Forecast ────────────────
    with tab2:
        st.subheader("🔮 Demand Forecasting — ML Model 1")
        st.markdown("Use trained Random Forest model to forecast revenue for any scenario.")

        col1, col2 = st.columns(2)
        with col1:
            f_cat    = st.selectbox("Medicine Category",
                sorted(df_sal['category'].unique()), key='f_cat')
            f_med    = st.selectbox("Medicine",
                sorted(df_sal[df_sal['category']==f_cat]
                       ['medicine_name'].unique()), key='f_med')
            f_month  = st.slider("Month", 1, 12, 7, key='f_month')
            f_qty    = st.slider("Expected Quantity", 1, 7, 3, key='f_qty')
            f_disc   = st.selectbox("Discount %", [0, 5, 10, 15], key='f_disc')

        with col2:
            if 'model1' in models:
                season_m = {1:'Winter',2:'Winter',3:'Summer',4:'Summer',
                            5:'Summer',6:'Monsoon',7:'Monsoon',8:'Monsoon',
                            9:'Monsoon',10:'Autumn',11:'Winter',12:'Winter'}
                f_season  = season_m[f_month]
                f_quarter = (f_month - 1) // 3 + 1
                avg_up    = df_sal[df_sal['medicine_name']==f_med]['unit_price'].mean()

                try:
                    cat_e = models['enc_cat'].transform([f_cat])[0]
                    med_e = models['enc_med'].transform([f_med])[0]
                    sea_e = models['enc_sea'].transform([f_season])[0]
                    Xf    = np.array([[f_month, f_quarter, sea_e,
                                       cat_e, med_e, avg_up, f_disc, f_qty]])
                    pred  = models['model1'].predict(Xf)[0]

                    month_names = {1:'January',2:'February',3:'March',
                                   4:'April',5:'May',6:'June',7:'July',
                                   8:'August',9:'September',10:'October',
                                   11:'November',12:'December'}
                    st.markdown(f"""<div class='pred-box'>
                        <div class='pred-lbl'>Forecasted Revenue</div>
                        <div class='pred-val'>₹{pred:,.0f}</div>
                        <div class='pred-lbl'>{f_med} · {month_names[f_month]} · {f_season}</div>
                    </div>""", unsafe_allow_html=True)

                    # Monthly forecast for all 12 months
                    st.markdown("**📅 12-Month Revenue Forecast**")
                    preds_12 = []
                    for m in range(1, 13):
                        s = season_m[m]; q = (m-1)//3+1
                        se = models['enc_sea'].transform([s])[0]
                        Xm = np.array([[m, q, se, cat_e, med_e,
                                        avg_up, f_disc, f_qty]])
                        preds_12.append({
                            'Month': list(month_names.values())[m-1],
                            'Forecasted Revenue': round(
                                models['model1'].predict(Xm)[0], 2)
                        })
                    df_12 = pd.DataFrame(preds_12)
                    fig_fc = px.bar(df_12, x='Month', y='Forecasted Revenue',
                                    color='Forecasted Revenue',
                                    color_continuous_scale='Blues')
                    fig_fc.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        height=260, margin=dict(l=10,r=10,t=10,b=10),
                        yaxis_tickprefix='₹', coloraxis_showscale=False)
                    st.plotly_chart(fig_fc, use_container_width=True)
                except Exception as e:
                    st.warning(f"Forecast error: {e}")
            else:
                st.warning("Model 1 not loaded.")

    # ── TAB 3: Geographic ─────────────────────
    with tab3:
        st.subheader("🌍 City-wise Revenue Analysis")
        city_sales = (df_sal.merge(df_cust[['customer_id','city']],
                                   on='customer_id', how='left')
                      .groupby('city')
                      .agg(revenue=('total_price','sum'),
                           transactions=('sale_id','count'))
                      .reset_index().sort_values('revenue', ascending=False))

        col1, col2 = st.columns(2)
        with col1:
            fig_city = px.bar(city_sales, x='city', y='revenue',
                              color='revenue',
                              color_continuous_scale='Blues',
                              labels={'revenue':'Revenue (₹)','city':'City'})
            fig_city.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                height=320, margin=dict(l=10,r=10,t=10,b=10),
                yaxis_tickprefix='₹', coloraxis_showscale=False)
            st.plotly_chart(fig_city, use_container_width=True)

        with col2:
            fig_city2 = px.pie(city_sales, values='revenue', names='city',
                               color_discrete_sequence=px.colors.qualitative.Set3,
                               hole=0.4)
            fig_city2.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=320,
                                    margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_city2, use_container_width=True)

        st.subheader("👥 Customer Demographics")
        col3, col4 = st.columns(2)
        with col3:
            df_cust['age_group'] = pd.cut(df_cust['age'],
                bins=[0,25,40,60,100],
                labels=['18–25','26–40','41–60','60+'])
            age_d = df_cust['age_group'].value_counts().sort_index().reset_index()
            age_d.columns = ['Age Group','Count']
            fig_age = px.bar(age_d, x='Age Group', y='Count',
                             color='Count', color_continuous_scale='Oranges')
            fig_age.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                height=260, margin=dict(l=10,r=10,t=10,b=10),
                coloraxis_showscale=False)
            st.plotly_chart(fig_age, use_container_width=True)

        with col4:
            gender_d = df_cust['gender'].value_counts().reset_index()
            gender_d.columns = ['Gender','Count']
            fig_gen = px.pie(gender_d, values='Count', names='Gender',
                             color_discrete_sequence=['#5B9BD5','#FF6B6B'],
                             hole=0.4)
            fig_gen.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=260,
                                  margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_gen, use_container_width=True)

    # ── TAB 4: Business Summary ───────────────
    with tab4:
        st.subheader("📈 Business Intelligence Summary")

        rev_2023 = df_sal[df_sal['year']==2023]['total_price'].sum()
        rev_2024 = df_sal[df_sal['year']==2024]['total_price'].sum()
        growth   = (rev_2024 - rev_2023) / rev_2023 * 100
        top_cat  = (df_sal.groupby('category')['total_price']
                    .sum().idxmax())
        top_med  = (df_sal.groupby('medicine_name')['total_price']
                    .sum().idxmax())
        best_mon = (df_sal.groupby(df_sal['date'].dt.to_period('M'))['total_price']
                    .sum().idxmax())

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-lbl'>2023 Revenue</div>
                <div class='metric-val'>₹{rev_2023/100000:.2f}L</div></div>""",
                unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class='metric-card metric-card-green'>
                <div class='metric-lbl'>2024 Revenue</div>
                <div class='metric-val'>₹{rev_2024/100000:.2f}L</div></div>""",
                unsafe_allow_html=True)
        with col3:
            color = 'green' if growth > 0 else 'red'
            st.markdown(f"""<div class='metric-card metric-card-{color}'>
                <div class='metric-lbl'>YoY Growth</div>
                <div class='metric-val'>{growth:+.1f}%</div></div>""",
                unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col4, col5 = st.columns(2)
        with col4:
            st.info(f"🏆 **Top Category:** {top_cat}")
            st.info(f"💊 **Top Medicine:** {top_med}")
            st.info(f"📅 **Best Month:** {best_mon}")
            st.info(f"👥 **Total Customers:** {len(df_cust):,}")

        with col5:
            critical_n = len(df_med[df_med['stock_status']=='Critical'])
            expired_n  = len(df_med[df_med['days_to_expiry'] < 0])
            expiring_n = len(df_med[(df_med['days_to_expiry']>=0) &
                                     (df_med['days_to_expiry']<=90)])
            avg_emp_perf = df_emp['performance_score'].mean()

            if critical_n > 0:
                st.error(f"🔴 {critical_n} medicine(s) need immediate reorder")
            else:
                st.success("✅ All stock levels are healthy")
            if expired_n > 0:
                st.error(f"🔴 {expired_n} medicine(s) have expired — remove from shelf")
            if expiring_n > 0:
                st.warning(f"⚠️ {expiring_n} medicine(s) expiring within 90 days")
            st.info(f"👤 Avg employee performance: **{avg_emp_perf:.1f}/100**")

        # Quarterly comparison
        st.subheader("📊 Quarterly Revenue Comparison")
        q_rev = (df_sal.groupby(['year','quarter'])['total_price']
                 .sum().reset_index())
        q_rev['Quarter'] = q_rev['quarter'].map(
            {1:'Q1',2:'Q2',3:'Q3',4:'Q4'})
        fig_q = px.bar(q_rev, x='Quarter', y='total_price',
                       color='year', barmode='group',
                       color_discrete_sequence=['#2563eb','#dc2626'],
                       labels={'total_price':'Revenue (₹)','year':'Year'})
        fig_q.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            height=300, margin=dict(l=10,r=10,t=10,b=10),
            yaxis_tickprefix='₹')
        st.plotly_chart(fig_q, use_container_width=True)