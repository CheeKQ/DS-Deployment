import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Garment Factory Productivity Predictor",
    page_icon="🧵",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.15rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 1.2rem;
    }
    .section-card {
        background-color: #f8fafc;
        padding: 1rem 1rem 0.8rem 1rem;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 1.2rem;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        background: #ffffff;
        margin-top: 0.8rem;
        margin-bottom: 1rem;
    }
    .small-note {
        color: #6b7280;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD ASSETS
# =========================================================
@st.cache_resource
def load_assets():
    model = joblib.load("garment_xgb_model.pkl")
    model_columns = joblib.load("xgb_model_columns.pkl")
    return model, model_columns

@st.cache_data
def load_reference_data():
    try:
        return pd.read_csv("cleaned_garments_worker_productivity.csv")
    except Exception:
        return None

model, model_columns = load_assets()
reference_df = load_reference_data()

# =========================================================
# SESSION STATE
# =========================================================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def get_range_from_data(df, column_name, default_min, default_max):
    if df is not None and column_name in df.columns:
        try:
            col = pd.to_numeric(df[column_name], errors="coerce").dropna()
            if not col.empty:
                return float(col.min()), float(col.max())
        except Exception:
            pass
    return float(default_min), float(default_max)

def set_dummy(category, value, input_df, columns):
    candidates = [
        f"{category}_{value}",
        f"{category}_{str(value).lower()}",
        f"{category}_{str(value).upper()}",
        f"{category}_{str(value).capitalize()}",
        f"{category}_{str(value).replace(' ', '_')}",
        f"{category}_{str(value).lower().replace(' ', '_')}"
    ]
    for col_name in candidates:
        if col_name in columns:
            input_df[col_name] = 1
            return

def build_recommendations(result, wip, workers, incentive, overtime, idle_time, idle_men, style_change):
    recs = []

    if result == "Low":
        recs.append("Reduce idle time and idle workers because these may negatively affect line efficiency.")
        recs.append("Review whether the current workload is too high for the available workforce.")
        recs.append("Consider improving incentive support to motivate better worker performance.")
        recs.append("Minimize unnecessary style changes to reduce disruption in production flow.")
        if overtime > 1.2:
            recs.append("High overtime may reflect operational pressure, so workload balancing should be reviewed.")
    elif result == "Moderate":
        recs.append("The production line is operating within a normal range but still has room for improvement.")
        recs.append("Better workload balancing and tighter floor control may help shift performance into the high tier.")
        recs.append("Keep idle time low and maintain stable worker allocation for more consistent productivity.")
        if style_change != "0":
            recs.append("Reducing style changes where possible may improve production continuity.")
    else:
        recs.append("The current setup appears efficient and supports strong productivity performance.")
        recs.append("This condition may be used as a benchmark for future production planning.")
        recs.append("Maintain current operational discipline and monitor for sudden changes in line conditions.")
        if idle_time == 0 and idle_men == 0:
            recs.append("Zero idle conditions suggest strong resource utilization and floor efficiency.")

    return recs

def normalize_prediction_label(pred_idx):
    labels = ["Low", "Moderate", "High"]
    try:
        return labels[int(pred_idx)]
    except Exception:
        return str(pred_idx)

# =========================================================
# DYNAMIC RANGES
# =========================================================
wip_min, wip_max = get_range_from_data(reference_df, "wip", 0, 23122)
workers_min, workers_max = get_range_from_data(reference_df, "no_of_workers", 2, 90)
smv_min, smv_max = get_range_from_data(reference_df, "smv", 2.9, 54.6)
incentive_min, incentive_max = get_range_from_data(reference_df, "incentive", 0, 3600)
idle_time_min, idle_time_max = get_range_from_data(reference_df, "idle_time", 0, 300)
idle_men_min, idle_men_max = get_range_from_data(reference_df, "idle_men", 0, 45)

wip_min = int(max(0, round(wip_min)))
wip_max = int(round(wip_max))
workers_min = int(max(2, round(workers_min)))
workers_max = int(round(workers_max))
incentive_min = int(max(0, round(incentive_min)))
incentive_max = int(round(incentive_max))
idle_time_min = int(max(0, round(idle_time_min)))
idle_time_max = int(round(idle_time_max))
idle_men_min = int(max(0, round(idle_men_min)))
idle_men_max = int(round(idle_men_max))

# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="main-title">🧵 AI-Powered Garment Factory Productivity Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">A decision support prototype for forecasting garment factory productivity using a tuned XGBoost classifier.</div>',
    unsafe_allow_html=True
)

st.success("✅ System Status: Model loaded successfully and ready for prediction.")

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("📌 Prototype Overview")
    st.write("""
This prototype predicts garment factory productivity into three classes:

- **Low**
- **Moderate**
- **High**

It uses a **tuned XGBoost classifier** trained on production-related features such as workload, labor allocation, incentives, overtime, and idle conditions.
""")

    st.markdown("---")
    st.subheader("🤖 Why XGBoost?")
    st.write("""
XGBoost was selected because it:
- captures complex non-linear relationships,
- provides strong predictive performance,
- reduces overfitting through regularization,
- performs well on structured operational data.
""")

    st.markdown("---")
    st.subheader("🧪 Current Model Inputs")
    st.write("""
- Day
- Quarter
- Department
- Team
- WIP
- Number of Workers
- Style Change
- SMV
- Incentive
- Overtime (Scaled)
- Idle Time
- Idle Workers
""")

# =========================================================
# INPUT AREA
# =========================================================
st.markdown("### 📥 Production Input Form")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📅 Time & Place")
    day = st.selectbox(
        "Day of the Week",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"]
    )
    quarter = st.selectbox(
        "Quarter",
        ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"]
    )
    dept = st.selectbox("Department", ["Sewing", "Finished"])
    team = st.slider("Team Number", 1, 12, 1)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("⚙️ Resource Allocation")
    wip = st.number_input(
        "Work in Progress (WIP)",
        min_value=0,
        max_value=max(wip_max, 1),
        value=min(500, max(wip_max, 1)),
        step=1
    )
    workers = st.number_input(
        "Number of Workers",
        min_value=workers_min,
        max_value=max(workers_max, workers_min),
        value=min(max(workers_min, 2), max(workers_max, workers_min)),
        step=1
    )
    style_change = st.selectbox("Number of Style Changes", ["0", "1", "2"])
    smv = st.number_input(
        "SMV (Complexity)",
        min_value=float(smv_min),
        max_value=float(smv_max),
        value=float(min(max(smv_min, 22.0), smv_max)),
        step=0.1,
        format="%.2f"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("💰 Incentives & Metrics")
    incentive = st.number_input(
        "Incentive Amount",
        min_value=0,
        max_value=max(incentive_max, 1),
        value=min(100, max(incentive_max, 1)),
        step=1
    )
    overtime = st.slider(
        "Overtime (Scaled)",
        min_value=-2.0,
        max_value=2.0,
        value=0.0,
        step=0.1
    )
    idle_time = st.number_input(
        "Idle Time (Mins)",
        min_value=0,
        max_value=max(idle_time_max, 1),
        value=0,
        step=1
    )
    idle_men = st.number_input(
        "Idle Workers Count",
        min_value=0,
        max_value=max(idle_men_max, 1),
        value=0,
        step=1
    )
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# INPUT MONITORING NOTES
# =========================================================
notes = []

if workers < 5:
    notes.append("Very low worker count may represent limited production capacity.")
if overtime > 1.5:
    notes.append("High overtime may indicate operational pressure.")
if idle_time > 180:
    notes.append("Idle time is unusually high and may reduce productivity.")
if idle_men > 20:
    notes.append("A high number of idle workers may indicate weak labor utilization.")
if wip > (0.8 * max(wip_max, 1)):
    notes.append("WIP is near the upper reference range and may create bottlenecks.")
if float(smv) > (0.8 * max(smv_max, 1)):
    notes.append("SMV is relatively high, which may indicate more complex production tasks.")

if notes:
    with st.expander("⚠️ Input Monitoring Notes"):
        for msg in notes:
            st.write(f"- {msg}")

# =========================================================
# PREDICTION
# =========================================================
st.divider()

if st.button("Generate Productivity Forecast", use_container_width=True):
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)

    # Numeric mapping
    numeric_map = {
        "team": team,
        "smv": float(smv),
        "wip": int(wip),
        "incentive": int(incentive),
        "idle_time": int(idle_time),
        "idle_men": int(idle_men),
        "no_of_workers": int(workers),
        "over_time_scaled": float(overtime)
    }

    for col, val in numeric_map.items():
        if col in input_df.columns:
            input_df[col] = val

    # Dummy encoding
    set_dummy("quarter", quarter, input_df, model_columns)
    set_dummy("department", dept, input_df, model_columns)
    set_dummy("department", dept.lower(), input_df, model_columns)
    set_dummy("day", day, input_df, model_columns)
    set_dummy("no_of_style_change", style_change, input_df, model_columns)

    input_df = input_df[model_columns]

    prediction_idx = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]

    result = normalize_prediction_label(prediction_idx)
    confidence = float(np.max(probs))

    history_row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Predicted Tier": result,
        "Confidence": f"{confidence:.2%}",
        "Day": day,
        "Quarter": quarter,
        "Department": dept,
        "Team": team,
        "WIP": int(wip),
        "Workers": int(workers),
        "Style Change": style_change,
        "SMV": float(smv),
        "Incentive": int(incentive),
        "Overtime Scaled": float(overtime),
        "Idle Time": int(idle_time),
        "Idle Workers": int(idle_men)
    }
    st.session_state.history.append(history_row)

    # =====================================================
    # RESULT METRICS
    # =====================================================
    st.markdown("## 📊 Prediction Results")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted Productivity", result)
    m2.metric("Confidence Score", f"{confidence:.2%}")
    m3.metric("Workers", int(workers))
    m4.metric("WIP", int(wip))

    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    if result == "High":
        st.success(f"✅ Predicted Tier: **{result}**")
        st.info("The model suggests that the current production setup is operating efficiently.")
    elif result == "Moderate":
        st.warning(f"⚠️ Predicted Tier: **{result}**")
        st.info("The model suggests that the current production setup is stable but still has room for improvement.")
    else:
        st.error(f"🚨 Predicted Tier: **{result}**")
        st.info("The model suggests a higher risk of lower productivity under the current operational setup.")
    st.markdown('</div>', unsafe_allow_html=True)

    # =====================================================
    # TABS
    # =====================================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📌 Input Summary",
        "📈 Confidence Breakdown",
        "💡 Recommendations",
        "🤖 XGBoost Explanation"
    ])

    with tab1:
        summary_df = pd.DataFrame({
            "Feature": [
                "Day",
                "Quarter",
                "Department",
                "Team Number",
                "Work in Progress (WIP)",
                "Number of Workers",
                "Style Changes",
                "SMV",
                "Incentive",
                "Overtime (Scaled)",
                "Idle Time",
                "Idle Workers"
            ],
            "Value": [
                day,
                quarter,
                dept,
                team,
                wip,
                workers,
                style_change,
                smv,
                incentive,
                overtime,
                idle_time,
                idle_men
            ]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    with tab2:
        prob_df = pd.DataFrame({
            "Productivity Level": ["Low", "Moderate", "High"],
            "Probability": probs
        })

        st.write("This chart shows the probability distribution across all productivity classes.")
        st.bar_chart(prob_df.set_index("Productivity Level"))

        display_df = prob_df.copy()
        display_df["Probability"] = display_df["Probability"].map(lambda x: f"{x:.2%}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    with tab3:
        st.write("The prototype also provides decision-support recommendations based on the predicted result.")
        recommendations = build_recommendations(
            result, wip, workers, incentive, overtime, idle_time, idle_men, style_change
        )
        for i, rec in enumerate(recommendations, start=1):
            st.write(f"**{i}.** {rec}")

    with tab4:
        st.write("""
**XGBoost** stands for **Extreme Gradient Boosting**.

It is an ensemble learning algorithm that combines many decision trees to improve prediction accuracy.

### Why XGBoost is suitable for this prototype
- It can capture complex relationships between production variables.
- It performs better than a single decision tree because it learns from previous prediction errors.
- It includes regularization to reduce overfitting.
- It is highly effective for structured production and workforce data.

### In this prototype
The model analyzes production-related variables such as:
- Day
- Quarter
- Department
- Team
- Work in Progress (WIP)
- Number of Workers
- Number of Style Changes
- SMV
- Incentive Amount
- Overtime (Scaled)
- Idle Time
- Idle Workers

Then, it predicts one of three productivity classes:
- **Low**
- **Moderate**
- **High**
""")

    # =====================================================
    # DOWNLOAD CURRENT RESULT
    # =====================================================
    st.markdown("### 📥 Export Current Result")
    export_df = pd.DataFrame([history_row])
    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Current Prediction as CSV",
        data=csv,
        file_name="garment_prediction_result.csv",
        mime="text/csv"
    )

# =========================================================
# HISTORY SECTION
# =========================================================
if st.session_state.history:
    st.divider()
    st.markdown("## 🕘 Prediction History")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True, hide_index=True)

    history_csv = history_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Full Prediction History",
        data=history_csv,
        file_name="garment_prediction_history.csv",
        mime="text/csv"
    )

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    '<div class="small-note">Prototype purpose: To support production planning, labor monitoring, and productivity forecasting in garment factory operations.</div>',
    unsafe_allow_html=True
)
