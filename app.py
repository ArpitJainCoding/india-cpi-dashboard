# ==========================================================
# INDIA MACRO-INFLATION DASHBOARD — OPTION A IMPLEMENTATION
# Uses real/backcast data only until 2026 and forecasts 2027–2060
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from pmdarima import auto_arima
from prophet import Prophet
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# ==========================================================
# GLOBAL SETTINGS
# ==========================================================
CURRENT_YEAR = 2026
FORECAST_END_YEAR = 2060
FORECAST_YEARS = FORECAST_END_YEAR - CURRENT_YEAR  # = 34 years

st.set_page_config(
    page_title="India Macro-Inflation Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🇮🇳 India Macro-Inflation Intelligence Dashboard (2000–2060)")
st.markdown("""
This dashboard shows inflation trends using MOSPI-anchored CPI category data (2000–2026)  
and **forecasts future inflation up to 2060 using ARIMA / Prophet / SES models**.
""")

# ==========================================================
# LOAD DATA
# ==========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("india_cpi_2000_2045_synthetic.csv")
    return df

df_raw = load_data()

# Trim dataset at 2026 (OPTION A)
df = df_raw[df_raw["year"] <= CURRENT_YEAR].copy()

categories = sorted(df["category"].unique())

# ==========================================================
# FORECAST FUNCTION
# ==========================================================
def generate_forecast(df, category, model_type="ARIMA"):
    sub = df[df["category"] == category].copy()
    sub = sub[sub["year"] <= CURRENT_YEAR]  # only real/backcast data
    sub = sub[["year", "pct_change_since_2000"]].reset_index(drop=True)
    sub.rename(columns={"pct_change_since_2000": "value"}, inplace=True)

    future_years = np.arange(CURRENT_YEAR + 1, FORECAST_END_YEAR + 1)

    if model_type == "ARIMA":
        model = auto_arima(sub["value"], seasonal=False, error_action="ignore")
        fc = model.predict(len(future_years))
        return pd.DataFrame({"year": future_years, "forecast": fc})

    elif model_type == "Prophet":
        dfp = sub.rename(columns={"year": "ds", "value": "y"})
        dfp["ds"] = pd.to_datetime(dfp["ds"], format="%Y")
        m = Prophet()
        m.fit(dfp)

        future = pd.DataFrame({
            "ds": pd.to_datetime(future_years, format="%Y")
        })
        forecast = m.predict(future)

        return pd.DataFrame({
            "year": future_years,
            "forecast": forecast["yhat"]
        })

    else:  # Simple Exponential Smoothing (SES)
        model = SimpleExpSmoothing(sub["value"]).fit()
        fc = model.forecast(len(future_years))
        return pd.DataFrame({"year": future_years, "forecast": fc})

# ==========================================================
# SIDEBAR CONTROLS
# ==========================================================
st.sidebar.header("Controls & Settings")

selected_categories = st.sidebar.multiselect(
    "Select CPI Categories",
    categories,
    default=categories
)

year_range = st.sidebar.slider(
    "Select Visible Year Range",
    min_value=2000,
    max_value=FORECAST_END_YEAR,
    value=(2000, FORECAST_END_YEAR)
)

show_yoy = st.sidebar.checkbox("Show YoY inflation", value=False)

st.sidebar.markdown("### 🔮 Forecast Settings")
enable_forecast = st.sidebar.checkbox("Enable Forecasting", value=True)
forecast_model = st.sidebar.selectbox("Forecast Model", ["ARIMA", "Prophet", "Simple"])

# ==========================================================
# PREPARE DATA
# ==========================================================
df_plot = df.copy()

if show_yoy:
    df_plot["yoy"] = df_plot.groupby("category")["pct_change_since_2000"].diff().fillna(0)

metric = "yoy" if show_yoy else "pct_change_since_2000"
ylabel = "YoY Inflation (%)" if show_yoy else "Cumulative % Change vs 2000"

# ==========================================================
# TABS
# ==========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Category Explorer",
    "📊 Services vs Goods",
    "🔥 Heatmap",
    "🧮 CPI Decomposition",
    "📉 Data"
])

# ==========================================================
# TAB 1 — CATEGORY EXPLORER WITH FORECAST
# ==========================================================
with tab1:
    st.subheader("📈 Category Trend + Forecast (2027–2060)")

    fig = go.Figure()

    for cat in selected_categories:

        sub = df_plot[df_plot["category"] == cat]

        # Actual Line
        fig.add_trace(go.Scatter(
            x=sub["year"],
            y=sub[metric],
            mode="lines+markers",
            name=f"{cat} — Actual"
        ))

        # Forecast Line
        if enable_forecast:
            fc = generate_forecast(df_plot, cat, model_type=forecast_model)
            fig.add_trace(go.Scatter(
                x=fc["year"],
                y=fc["forecast"],
                mode="lines",
                line=dict(dash="dash"),
                name=f"{cat} — Forecast"
            ))

    # Highlight CURRENT YEAR (2026)
    fig.add_vline(
        x=CURRENT_YEAR,
        line_width=3,
        line_dash="dot",
        line_color="red",
        annotation_text=f"Current Year: {CURRENT_YEAR}",
        annotation_position="top"
    )

    # Highlight datapoints at CURRENT YEAR
    current_points = df_plot[df_plot["year"] == CURRENT_YEAR]
    fig.add_trace(go.Scatter(
        x=current_points["year"],
        y=current_points[metric],
        mode="markers",
        marker=dict(size=12, color="red"),
        name="Current Year Value"
    ))

    fig.update_layout(
        height=600,
        xaxis_title="Year",
        yaxis_title=ylabel,
        title="Inflation Trend (Actual Up to 2026) + Forecast (2027–2060)"
    )

    # Restrict chart range
    fig.update_xaxes(range=[year_range[0], year_range[1]])

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"""
        <div style='padding:10px; background:#eef; border-radius:8px; text-align:center;'>
            📌 <b>Timeline Note:</b> Historical data shown until <b>{CURRENT_YEAR}</b>.  
            Forecasts shown for <b>{CURRENT_YEAR+1}–{FORECAST_END_YEAR}</b>.
        </div>
        """,
        unsafe_allow_html=True
    )

# ==========================================================
# TAB 2 — SERVICES VS GOODS
# ==========================================================
with tab2:
    st.subheader("📊 Services vs Goods Inflation")

    services = ["Health", "Education", "Personal care & effects", "Miscellaneous services", "Housing"]
    goods = ["Food & beverages", "Clothing & footwear", "Fuel & light",
             "Household goods & services", "Transport & communication"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Service Sector")
        df_services = df_plot[df_plot["category"].isin(services)]
        fig_s = px.line(df_services, x="year", y=metric, color="category",
                        title="Services (Actual Until 2026)")
        st.plotly_chart(fig_s, use_container_width=True)

    with col2:
        st.markdown("### Goods Sector")
        df_goods = df_plot[df_plot["category"].isin(goods)]
        fig_g = px.line(df_goods, x="year", y=metric, color="category",
                        title="Goods (Actual Until 2026)")
        st.plotly_chart(fig_g, use_container_width=True)

# ==========================================================
# TAB 3 — HEATMAP
# ==========================================================
with tab3:
    st.subheader("🔥 Inflation Heatmap (2000–2026 Only)")

    heat_data = df_plot.pivot_table(
        index="category", columns="year", values="pct_change_since_2000"
    )

    fig_h = px.imshow(
        heat_data,
        aspect="auto",
        color_continuous_scale="Turbo",
        title="Cumulative % Change vs 2000"
    )

    st.plotly_chart(fig_h, use_container_width=True)

# ==========================================================
# TAB 4 — CPI DECOMPOSITION
# ==========================================================
with tab4:
    st.subheader("🧮 CPI Decomposition")

    fig_d = px.area(
        df_plot,
        x="year",
        y="pct_change_since_2000",
        color="category",
        groupnorm="fraction",
        title="Category Share of Total CPI Movement"
    )

    st.plotly_chart(fig_d, use_container_width=True)

# ==========================================================
# TAB 5 — DATA VIEWER
# ==========================================================
with tab5:
    st.subheader("📉 Dataset Used (Trimmed to 2000–2026)")
    st.dataframe(df_plot, use_container_width=True)

    st.download_button(
        "Download CSV",
        df_plot.to_csv(index=False),
        "cpi_2000_2026_trimmed.csv",
        "text/csv"
    )
