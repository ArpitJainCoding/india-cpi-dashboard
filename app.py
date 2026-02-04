import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pmdarima import auto_arima
from prophet import Prophet

def generate_forecast(df, category, horizon=30, model_type="ARIMA"):
    sub = df[df["category"] == category].copy()
    sub = sub[["year", "pct_change_since_2000"]].reset_index(drop=True)
    sub.rename(columns={"pct_change_since_2000": "value"}, inplace=True)

    # Forecast years
    future_years = np.arange(sub["year"].max() + 1, sub["year"].max() + horizon + 1)

    if model_type == "ARIMA":
        model = auto_arima(sub["value"], seasonal=False, error_action="ignore")
        fc = model.predict(horizon)
        return pd.DataFrame({"year": future_years, "forecast": fc})

    elif model_type == "Prophet":
        dfp = sub.rename(columns={"year": "ds", "value": "y"})
        dfp["ds"] = pd.to_datetime(dfp["ds"], format="%Y")
        m = Prophet()
        m.fit(dfp)

        future = m.make_future_dataframe(periods=horizon, freq="Y")
        forecast = m.predict(future)
        fc = forecast.tail(horizon)

        return pd.DataFrame({
            "year": fc["ds"].dt.year,
            "forecast": fc["yhat"]
        })

    else:  # Simple exponential smoothing
        from statsmodels.tsa.holtwinters import SimpleExpSmoothing
        model = SimpleExpSmoothing(sub["value"]).fit()
        fc = model.forecast(horizon)
        return pd.DataFrame({"year": future_years, "forecast": fc})


# -----------------------------------------
# CONFIG
# -----------------------------------------
st.set_page_config(
    page_title="India Macro-Inflation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🇮🇳 India Macro-Inflation Intelligence Dashboard (2000–2045)")
st.markdown("""
This interactive dashboard visualizes cumulative inflation paths for major CPI-style categories in India
from **2000 to 2045**, using MOSPI-anchored historical data (2012–2025) and projection assumptions (2026–2045).
""")

# -----------------------------------------
# LOAD DATA
# -----------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("india_cpi_2000_2045_synthetic.csv")
    return df

df = load_data()

categories = sorted(df["category"].unique())

# -----------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------
st.sidebar.header("Filters & Controls")
selected_categories = st.sidebar.multiselect("Select Categories", categories, default=categories)

year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=int(df.year.min()),
    max_value=int(df.year.max()),
    value=(2000, 2045)
)

show_yoy = st.sidebar.checkbox("Show YoY Inflation Instead of Cumulative %", value=False)



st.sidebar.markdown("### 🔮 Forecasting Options")
enable_forecast = st.sidebar.checkbox("Enable Forecasting", value=True)
forecast_model = st.sidebar.selectbox("Forecast Model", ["ARIMA", "Prophet", "Simple"])
forecast_horizon = st.sidebar.slider("Forecast Horizon (Years)", 5, 30, 20)


# Scenario adjustments
st.sidebar.subheader("Projection Scenario (2026–2045)")
inflation_adj = st.sidebar.slider(
    "Adjust Future Inflation (%)",
    min_value=-3.0, max_value=5.0, value=0.0, step=0.1,
    help="Applies an additive adjustment to projected inflation rates."
)

# -----------------------------------------
# PROCESS DATA (Scenario Engine)
# -----------------------------------------
df_plot = df.copy()

# Calculate YoY inflation if needed
if show_yoy:
    df_plot["yoy"] = df_plot.groupby("category")["pct_change_since_2000"].diff().fillna(0)

# Apply scenario adjustment only to projected years
future_mask = df_plot["year"] >= 2026
df_plot.loc[future_mask, "pct_change_since_2000"] *= (1 + inflation_adj/100)
if show_yoy:
    df_plot.loc[future_mask, "yoy"] *= (1 + inflation_adj/100)

df_display = df_plot[
    (df_plot["year"] >= year_range[0]) &
    (df_plot["year"] <= year_range[1]) &
    (df_plot["category"].isin(selected_categories))
]

# -----------------------------------------
# MAIN TABS
# -----------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Category Explorer",
    "📊 Services vs Goods",
    "🔥 Inflation Heatmap",
    "🧮 CPI Decomposition",
    "📉 Raw Data"
])

# -----------------------------------------
# TAB 1 — Category Explorer
# -----------------------------------------

with tab1:
    st.subheader("📈 Category-Level Inflation Trend with Forecasting")

    metric = "yoy" if show_yoy else "pct_change_since_2000"
    ylabel = "YoY Inflation (%)" if show_yoy else "Cumulative % Change vs 2000"

    fig = go.Figure()

    # Plot actual data
    for cat in selected_categories:
        sub = df_display[df_display["category"] == cat]
        fig.add_trace(go.Scatter(
            x=sub["year"],
            y=sub[metric],
            mode="lines+markers",
            name=f"{cat} – Actual"
        ))

        # Forecast overlay
        if enable_forecast:
            fc = generate_forecast(df_plot, cat, forecast_horizon, forecast_model)
            fig.add_trace(go.Scatter(
                x=fc["year"],
                y=fc["forecast"],
                mode="lines",
                line=dict(dash="dash"),
                name=f"{cat} – Forecast"
            ))

    # Highlight current year
    current_year = df["year"].max()
    fig.add_vline(
        x=current_year,
        line_width=3,
        line_dash="dot",
        line_color="red",
        annotation_text=f"Current Year: {current_year}",
        annotation_position="top"
    )

    # Highlight datapoints at current year
    current_points = df_plot[df_plot["year"] == current_year]
    fig.add_trace(go.Scatter(
        x=current_points["year"],
        y=current_points[metric],
        mode="markers",
        marker=dict(size=12, color="red"),
        name="Current Year Points"
    ))

    fig.update_layout(
        height=600,
        title="Inflation Trend + Forecast",
        xaxis_title="Year",
        yaxis_title=ylabel,
    )

    st.plotly_chart(fig, use_container_width=True)


# -----
    st.markdown(
    f"""
    <div style='padding:10px; background:#eef; border-radius:8px; text-align:center;'>
        <b>📌 Timeline Marker:</b> Showing actual data through <span style='color:red'>{current_year}</span> and forecast thereafter.
    </div>
    """,
    unsafe_allow_html=True
)
#-----------------------------------
# TAB 2 — Services vs Goods Split
# -----------------------------------------
with tab2:
    st.subheader("📊 Services vs Goods Inflation Divergence")

    services = ["Health", "Education", "Personal care & effects", "Miscellaneous services", "Housing"]
    goods = ["Food & beverages", "Clothing & footwear", "Fuel & light",
             "Household goods & services", "Transport & communication"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Service Categories")
        df_services = df_display[df_display["category"].isin(services)]
        fig_s = px.line(df_services, x="year", y=metric, color="category",
                        title="Service Sector Inflation")
        st.plotly_chart(fig_s, use_container_width=True)

    with col2:
        st.markdown("### Goods Categories")
        df_goods = df_display[df_display["category"].isin(goods)]
        fig_g = px.line(df_goods, x="year", y=metric, color="category",
                        title="Goods Sector Inflation")
        st.plotly_chart(fig_g, use_container_width=True)

# -----------------------------------------
# TAB 3 — Inflation Heatmap
# -----------------------------------------
with tab3:
    st.subheader("🔥 Inflation Velocity Heatmap")

    heat_data = df_plot.pivot_table(
        index="category", columns="year", values="pct_change_since_2000"
    )
    fig_h = px.imshow(
        heat_data,
        aspect="auto",
        color_continuous_scale="Turbo",
        title="Inflation Heatmap (Cumulative % vs 2000)"
    )
    st.plotly_chart(fig_h, use_container_width=True)

# -----------------------------------------
# TAB 4 — CPI Decomposition
# -----------------------------------------
with tab4:
    st.subheader("🧮 CPI Decomposition Tool")

    # Assume equal weights unless user wants custom (can be added later)
    df_decomp = df_plot[df_plot["category"].isin(categories)]
    df_decomp = df_decomp.groupby("year")["pct_change_since_2000"].mean().reset_index()

    fig_d = px.area(
        df_display,
        x="year",
        y="pct_change_since_2000",
        color="category",
        groupnorm="fraction",
        title="Share of Cumulative Inflation by Category"
    )
    fig_d.update_layout(height=550)
    st.plotly_chart(fig_d, use_container_width=True)

# -----------------------------------------
# TAB 5 — Raw Data Viewer
# -----------------------------------------
with tab5:
    st.subheader("📉 Raw Dataset")
    st.dataframe(df_plot, use_container_width=True)

    csv = df_plot.to_csv(index=False).encode("utf-8")
    st.download_button("Download Filtered Data as CSV", csv, "filtered_cpi_data.csv", "text/csv")

# -----------------------------------------
# END
# -----------------------------------------
st.success("Dashboard loaded successfully. Adjust sidebar filters to explore inflation dynamics!")
