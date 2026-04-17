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
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 1.2rem;
    }
    .section-card {
        background-color: #f8fafc;
        padding: 1rem 1rem 0.6rem 1rem;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 1.2rem;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        background: #ffffff;
        margin-top: 1rem;
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
def load_model_assets():
    model = joblib.load("garment_xgb_model.pkl")
    model_columns = joblib.load("xgb_model_columns.pkl")
    return model, model_columns

@st.cache_data
def load_reference_data():
    try:
        df = pd.read_csv("cleaned_garments_worker_productivity.csv")
        return df
    except Exception:
        return None

model, model_columns = load_model_assets()
reference_df = load_reference_data()

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def get_dynamic_range(df, column_name, default_min, default_max):
    if df is not None and column_name in df.columns:
        try:
            col = pd.to_numeric(df[column_name], errors="coerce").dropna()
            if len(col) > 0:
                return float(col.min()), float(col.max())
        except Exception:
            pass
    return float(default_min), float(default_max)

def set_matching_dummy(input_df, prefix, value, columns):
    """
    Tries several possible dummy column formats to improve compatibility
    with training columns.
    """
    candidates = [
        f"{prefix}_{value}",
        f"{prefix}_{str(value).lower()}",
        f"{prefix}_{str(value).upper()}",
        f"{prefix}_{str(value).capitalize()}",
        f"{prefix}_{str(value).replace(' ', '_')}",
        f"{prefix}_{str(value).lower().replace(' ', '_')}",
        f"{prefix}_{str(value).upper().replace(' ', '_')}",
    ]
    for col in candidates:
        if col in columns:
            input_df[col] = 1
            return

def build_recommendations(result, wip, workers, incentive, overtime, idle_time, idle_men, style_change):
    recs = []

    if result == "Low":
        recs.append("Reduce idle time and idle workers because these factors may lower production efficiency.")
        recs.append("Review workload distribution to ensure the current WIP is manageable for the available workforce.")
        recs.append("Consider adjusting incentive strategy to improve worker motivation and output.")
        recs.append("Monitor whether excessive operational disruptions, such as style changes, are affecting stability.")
        if overtime > 1.2:
            recs.append("High overtime may indicate pressure on operations, so workload balancing should be reviewed.")
    elif result == "Moderate":
        recs.append("The production line is operating within a normal range, but there is still room for improvement.")
        recs.append("Fine-tune worker allocation and monitor WIP to push performance toward the high-productivity tier.")
        recs.append("Keep idle time low and maintain stable floor conditions to improve consistency.")
        if style_change != "0":
            recs.append("Reducing style changes where possible may improve operational continuity.")
    else:
        recs.append("The current setup appears efficient and supports strong productivity performance.")
        recs.append("Maintain the present resource allocation and continue monitoring for sudden disruptions.")
        recs.append("This scenario may be used as a benchmark for future production planning.")
        if idle_time == 0 and idle_men == 0:
            recs.append("Very low idle conditions suggest strong operational discipline and resource utilization.")

    return recs

def get_result_color(result):
    if result == "High":
        return "green"
    elif result == "Moderate":
        return "orange"
    return "red"

def normalize_prediction_label(pred_idx):
    labels = ["Low", "Moderate", "High"]
    try:
        return labels[int(pred_idx)]
    except Exception:
        return str(pred_idx)

# =========================================================
# SESSION STATE
# =========================================================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="main-title">🧵 AI-Powered Garment Factory Productivity Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">A machine learning-based decision support system for forecasting garment factory productivity using XGBoost.</div>',
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

It uses a **Tuned XGBoost Classifier** trained on operational variables such as:
- Work in Progress (WIP)
- Number of workers
- Incentive amount
- Idle time
- Overtime
- Style change frequency
- Department and time context
""")

    st.markdown("---")
    st.subheader("🤖 Why XGBoost?")
    st.write("""
XGBoost was selected because it:
- captures non-linear relationships,
- improves predictive performance,
- reduces overfitting through regularization,
- works well for structured production data.
""")

    st.markdown("---")
    st.subheader("🧪 Model Inputs")
    st.caption(f"Expected feature columns: **{len(model_columns)}**")

# =========================================================
# DYNAMIC RANGES
# =========================================================
wip_min, wip_max = get_dynamic_range(reference_df, "wip", 0, 23122)
workers_min, workers_max = get_dynamic_range(reference_df, "no_of_workers", 2, 90)
smv_min, smv_max = get_dynamic_range(reference_df, "smv", 2.9, 54.6)
incentive_min, incentive_max = get_dynamic_range(reference_df, "incentive", 0, 3600)
idle_time_min, idle_time_max = get_dynamic_range(reference_df, "idle_time", 0, 300)
idle_men_min, idle_men_max = get_dynamic_range(reference_df, "idle_men", 0, 45)

# Make ranges presentation-safe
wip_min = int(max(0, round(wip_min)))
wip_max = int(round(wip_max))
workers_min = int(max(1, round(workers_min)))
workers_max = int(round(workers_max))
incentive_min = int(max(0, round(incentive_min)))
incentive_max = int(round(incentive_max))
idle_time_min = int(max(0, round(idle_time_min)))
idle_time_max = int(round(idle_time_max))
idle_men_min = int(max(0, round(idle_men_min)))
idle_men_max = int(round(idle_men_max))

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
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
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
    wip = st.number_input("Work in Progress (WIP)", min_value=0, max_value=max(wip_max, 1), value=min(500, max(wip_max, 1)), step=1)
    workers = st.number_input("Number of Workers", min_value=max(workers_min, 1), max_value=max(workers_max, 2), value=min(max(workers_min, 2), max(workers_max, 2)))
    style_change = st.selectbox("Number of Style Changes", ["0", "1", "2"])
    smv = st.number_input("SMV (Complexity)", min_value=float(smv_min), max_value=float(smv_max), value=float(min(max(smv_min, 22.0), smv_max)), step=0.1, format="%.2f")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("💰 Incentives & Metrics")
    incentive = st.number_input("Incentive Amount", min_value=0, max_value=max(incentive_max, 1), value=min(100, max(incentive_max, 1)), step=1)
    overtime = st.slider("Overtime (Scaled)", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
    idle_time = st.number_input("Idle Time (Mins)", min_value=0, max_value=max(idle_time_max, 1), value=0, step=1)
    idle_men = st.number_input("Idle Workers Count", min_value=0, max_value=max(idle_men_max, 1), value=0, step=1)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# VALIDATION MESSAGES
# =========================================================
warnings = []

if workers < 2:
    warnings.append("Number of workers is very low and may not reflect normal production conditions.")
if overtime > 1.5:
    warnings.append("High overtime may indicate operational pressure and should be interpreted carefully.")
if idle_time > 180:
    warnings.append("Idle time is unusually high and may strongly reduce productivity.")
if idle_men > 20:
    warnings.append("A high number of idle workers may signal poor resource utilization.")
if wip > (0.8 * wip_max):
    warnings.append("WIP is near the upper reference range and may create production bottlenecks.")
if float(smv) > (0.8 * smv_max):
    warnings.append("SMV is relatively high, which may indicate more complex production tasks.")

if warnings:
    with st.expander("⚠️ Input Monitoring Notes"):
        for msg in warnings:
            st.write(f"- {msg}")

# =========================================================
# PREDICTION
# =========================================================
st.divider()

predict_clicked = st.button("Generate Productivity Forecast", use_container_width=True)

if predict_clicked:
    # Build input frame
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)

    # Numeric fields
    numeric_map = {
        "team": team,
        "smv": float(smv),
        "wip": int(wip),
        "incentive": int(incentive),
        "idle_time": int(idle_time),
        "idle_men": int(idle_men),
        "no_of_workers": int(workers),
        "over_time_scaled": float(overtime),
    }

    for col, val in numeric_map.items():
        if col in input_df.columns:
            input_df[col] = val

    # Dummy encoding
    set_matching_dummy(input_df, "quarter", quarter, model_columns)
    set_matching_dummy(input_df, "department", dept, model_columns)
    set_matching_dummy(input_df, "department", dept.lower(), model_columns)
    set_matching_dummy(input_df, "day", day, model_columns)
    set_matching_dummy(input_df, "no_of_style_change", style_change, model_columns)

    # Align order
    input_df = input_df[model_columns]

    # Predict
    prediction_idx = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]

    result = normalize_prediction_label(prediction_idx)
    confidence = float(np.max(probs))
    result_color = get_result_color(result)

    # Save history
    history_row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Predicted Tier": result,
        "Confidence": f"{confidence:.2%}",
        "WIP": wip,
        "Workers": workers,
        "Incentive": incentive,
        "Idle Time": idle_time,
        "Idle Workers": idle_men,
        "SMV": smv,
        "Overtime": overtime,
        "Department": dept,
        "Day": day,
        "Quarter": quarter
    }
    st.session_state.history.append(history_row)

    # =====================================================
    # RESULTS AREA
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

        st.dataframe(
            prob_df.assign(Probability=lambda x: x["Probability"].map(lambda y: f"{y:.2%}")),
            use_container_width=True,
            hide_index=True
        )

    with tab3:
        st.write("The prototype also provides decision-support recommendations based on the prediction result.")
        recommendations = build_recommendations(
            result, wip, workers, incentive, overtime, idle_time, idle_men, style_change
        )
        for i, rec in enumerate(recommendations, start=1):
            st.write(f"**{i}.** {rec}")

    with tab4:
        st.write("""
**XGBoost** stands for **Extreme Gradient Boosting**.  
It is an advanced ensemble learning algorithm that combines many decision trees to improve prediction accuracy.

### Why XGBoost is suitable for this prototype
- It can capture **complex and non-linear relationships** between production variables.
- It generally performs better than a single decision tree because it learns from previous prediction errors.
- It includes **regularization**, which helps reduce overfitting.
- It is highly effective for **structured tabular datasets** like garment factory operational data.

### In this prototype
The model analyzes production-related variables such as:
- Work in Progress (WIP)
- Number of workers
- Incentive amount
- Overtime
- Idle time
- Department
- Style changes
- SMV complexity

Then, it predicts one of three productivity classes:
- **Low**
- **Moderate**
- **High**

### How to explain during presentation
You can say:

> “This prototype uses a tuned XGBoost classifier to evaluate operational conditions in a garment factory and forecast the expected productivity level. Compared with a single model, XGBoost provides stronger predictive performance because it improves itself by learning from previous mistakes.”
""")

    # =====================================================
    # DOWNLOAD SECTION
    # =====================================================
    st.markdown("### 📥 Export Result")
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

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    '<div class="small-note">Prototype purpose: To support production planning, resource monitoring, and productivity forecasting in a garment factory environment.</div>',
    unsafe_allow_html=True
)
