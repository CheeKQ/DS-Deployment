import streamlit as st
import pandas as pd
import numpy as np
import joblib

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
        font-size: 2.5rem;
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
        padding: 1rem 1rem 0.8rem 1rem;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
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
def load_dataset():
    return pd.read_csv("final_classification_dataset.csv")

model, model_columns = load_model_assets()
df = load_dataset()

# =========================================================
# DATA-DRIVEN OPTIONS
# =========================================================
quarter_options = sorted(df["quarter"].dropna().unique().tolist())
department_options = sorted(df["department"].dropna().unique().tolist())
day_options = sorted(df["day"].dropna().unique().tolist())
style_change_options = sorted(df["no_of_style_change"].dropna().unique().tolist())

# numeric ranges
smv_min, smv_max = float(df["smv"].min()), float(df["smv"].max())
wip_min, wip_max = int(df["wip"].min()), int(df["wip"].max())
over_time_min, over_time_max = int(df["over_time"].min()), int(df["over_time"].max())
incentive_min, incentive_max = int(df["incentive"].min()), int(df["incentive"].max())
idle_time_min, idle_time_max = int(df["idle_time"].min()), int(df["idle_time"].max())
idle_men_min, idle_men_max = int(df["idle_men"].min()), int(df["idle_men"].max())
workers_min, workers_max = int(df["no_of_workers"].min()), int(df["no_of_workers"].max())

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def set_dummy_value(input_df, prefix, value):
    candidates = [
        f"{prefix}_{value}",
        f"{prefix}_{str(value).lower()}",
        f"{prefix}_{str(value).upper()}",
        f"{prefix}_{str(value).capitalize()}",
    ]
    for col in candidates:
        if col in input_df.columns:
            input_df[col] = 1
            return

def normalize_prediction(pred):
    if isinstance(pred, str):
        return pred
    class_map = {
        0: "Low",
        1: "Moderate",
        2: "High"
    }
    return class_map.get(int(pred), str(pred))

def get_result_message(result):
    if result == "High":
        st.balloons()
        return "success", "The current input pattern suggests strong production performance."
    elif result == "Moderate":
        return "warning", "The current input pattern suggests average but stable production performance."
    else:
        return "error", "The current input pattern suggests a risk of lower productivity."

def get_recommendations(result, wip, over_time, incentive, idle_time, idle_men, workers, style_change):
    recs = []

    if result == "Low":
        recs.append("Reduce idle time and idle workers to improve operational efficiency.")
        recs.append("Review whether the current workload is balanced with the number of workers.")
        recs.append("Evaluate incentive strategy to improve worker motivation and performance.")
        recs.append("Minimize unnecessary style changes to reduce disruption on the production floor.")
        if over_time > 7000:
            recs.append("Very high overtime may indicate pressure on the production line and should be monitored carefully.")

    elif result == "Moderate":
        recs.append("The production line is operating within a normal range but still has room for improvement.")
        recs.append("Better workload planning and lower idle conditions may help raise productivity to the high tier.")
        recs.append("Monitor overtime and style changes to maintain stable production flow.")

    else:
        recs.append("The current production setup appears efficient and well balanced.")
        recs.append("This input combination can be used as a benchmark for future planning.")
        recs.append("Maintain low idle conditions and consistent resource allocation to sustain high productivity.")

    return recs

# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="main-title">🧵 AI-Powered Garment Factory Productivity Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">A machine learning-based decision support prototype for predicting garment factory productivity using XGBoost.</div>',
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

The prediction is based on operational variables from the finalized classification dataset.
""")

    st.markdown("---")
    st.subheader("🤖 Why XGBoost?")
    st.write("""
XGBoost was chosen because it:
- handles non-linear relationships well,
- improves accuracy through boosting,
- reduces overfitting with regularization,
- performs strongly on structured tabular data.
""")

    st.markdown("---")
    st.subheader("📂 Dataset-Aligned Inputs")
    st.write("""
This app is built based on these finalized dataset fields:
- quarter
- department
- day
- smv
- wip
- over_time
- incentive
- idle_time
- idle_men
- no_of_style_change
- no_of_workers
""")

# =========================================================
# INPUT AREA
# =========================================================
st.markdown("## 📥 Production Input Form")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📅 Time & Context")
    quarter = st.selectbox("Quarter", quarter_options)
    department = st.selectbox("Department", department_options)
    day = st.selectbox("Day of the Week", day_options)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("⚙️ Production Factors")
    smv = st.number_input("SMV (Complexity)", min_value=smv_min, max_value=smv_max, value=float(round(df["smv"].median(), 2)), step=0.01, format="%.2f")
    wip = st.number_input("Work in Progress (WIP)", min_value=wip_min, max_value=wip_max, value=int(df["wip"].median()), step=1)
    no_of_style_change = st.selectbox("Number of Style Changes", style_change_options)
    no_of_workers = st.number_input("Number of Workers", min_value=workers_min, max_value=workers_max, value=int(df["no_of_workers"].median()), step=1)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("💰 Time & Efficiency Metrics")
    over_time = st.number_input("Overtime", min_value=over_time_min, max_value=over_time_max, value=int(df["over_time"].median()), step=1)
    incentive = st.number_input("Incentive Amount", min_value=incentive_min, max_value=incentive_max, value=int(df["incentive"].median()), step=1)
    idle_time = st.number_input("Idle Time (Mins)", min_value=idle_time_min, max_value=idle_time_max, value=int(df["idle_time"].median()), step=1)
    idle_men = st.number_input("Idle Workers Count", min_value=idle_men_min, max_value=idle_men_max, value=int(df["idle_men"].median()), step=1)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# MONITORING NOTES
# =========================================================
notes = []

if idle_time > 100:
    notes.append("Idle time is relatively high and may negatively affect productivity.")
if idle_men > 10:
    notes.append("A high number of idle workers may indicate weak labor utilization.")
if over_time > 8000:
    notes.append("Very high overtime may reflect operational pressure.")
if wip > 2000:
    notes.append("A high WIP value may create bottlenecks if not managed properly.")

if notes:
    with st.expander("⚠️ Input Monitoring Notes"):
        for note in notes:
            st.write(f"- {note}")

# =========================================================
# PREDICTION
# =========================================================
st.divider()

if st.button("Generate Productivity Forecast", use_container_width=True):
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)

    # numeric fields
    numeric_fields = {
        "smv": smv,
        "wip": wip,
        "over_time": over_time,
        "incentive": incentive,
        "idle_time": idle_time,
        "idle_men": idle_men,
        "no_of_style_change": no_of_style_change,
        "no_of_workers": no_of_workers
    }

    for col, val in numeric_fields.items():
        if col in input_df.columns:
            input_df[col] = val

    # dummy fields
    set_dummy_value(input_df, "quarter", quarter)
    set_dummy_value(input_df, "department", department)
    set_dummy_value(input_df, "day", day)

    # some models one-hot encode style changes
    set_dummy_value(input_df, "no_of_style_change", no_of_style_change)

    # align
    input_df = input_df[model_columns]

    # predict
    raw_pred = model.predict(input_df)[0]
    result = normalize_prediction(raw_pred)

    probs = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(input_df)[0]
        except Exception:
            probs = None

    status_type, status_msg = get_result_message(result)

    st.markdown("## 📊 Prediction Results")

    if probs is not None:
        confidence = float(np.max(probs))
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Productivity", result)
        c2.metric("Confidence Score", f"{confidence:.2%}")
        c3.metric("Workers", int(no_of_workers))
    else:
        c1, c2 = st.columns(2)
        c1.metric("Predicted Productivity", result)
        c2.metric("Workers", int(no_of_workers))

    if status_type == "success":
        st.success(status_msg)
    elif status_type == "warning":
        st.warning(status_msg)
    else:
        st.error(status_msg)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📌 Input Summary",
        "📈 Confidence Breakdown",
        "💡 Recommendations",
        "🤖 XGBoost Explanation"
    ])

    with tab1:
        summary_df = pd.DataFrame({
            "Feature": [
                "Quarter",
                "Department",
                "Day",
                "SMV",
                "WIP",
                "Overtime",
                "Incentive",
                "Idle Time",
                "Idle Workers",
                "Style Changes",
                "Number of Workers"
            ],
            "Value": [
                quarter,
                department,
                day,
                smv,
                wip,
                over_time,
                incentive,
                idle_time,
                idle_men,
                no_of_style_change,
                no_of_workers
            ]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    with tab2:
        if probs is not None:
            prob_df = pd.DataFrame({
                "Productivity Level": ["Low", "Moderate", "High"],
                "Probability": probs
            })
            st.bar_chart(prob_df.set_index("Productivity Level"))
            display_df = prob_df.copy()
            display_df["Probability"] = display_df["Probability"].map(lambda x: f"{x:.2%}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("Probability output is not available for the current loaded model.")

    with tab3:
        recommendations = get_recommendations(
            result, wip, over_time, incentive, idle_time, idle_men, no_of_workers, no_of_style_change
        )
        for i, rec in enumerate(recommendations, start=1):
            st.write(f"**{i}.** {rec}")

    with tab4:
        st.write("""
**XGBoost** stands for **Extreme Gradient Boosting**.

It is an ensemble learning algorithm that combines many decision trees to improve prediction accuracy.

### Why XGBoost is suitable for this prototype
- It can model complex relationships between production factors.
- It performs better than a single tree by learning from previous errors.
- It includes regularization to reduce overfitting.
- It is highly effective for structured operational datasets.

### In this prototype
The model analyzes:
- quarter
- department
- day
- smv
- wip
- over_time
- incentive
- idle_time
- idle_men
- no_of_style_change
- no_of_workers

Then, it predicts the productivity class as:
- **Low**
- **Moderate**
- **High**
""")
    # =====================================================
    # EXPORT CURRENT RESULT
    # =====================================================
    st.subheader("📥 Export Current Result")
    current_result_df = pd.DataFrame([history_row])
    current_csv = current_result_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Current Prediction as CSV",
        data=current_csv,
        file_name="current_prediction_result.csv",
        mime="text/csv"
    )

# =========================================================
# PREDICTION HISTORY
# =========================================================
if st.session_state.prediction_history:
    st.divider()
    st.subheader("🕘 Prediction History")

    history_df = pd.DataFrame(st.session_state.prediction_history)
    st.dataframe(history_df, use_container_width=True, hide_index=True)

    history_csv = history_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Full Prediction History",
        data=history_csv,
        file_name="prediction_history.csv",
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



