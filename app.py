import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from xgboost import XGBClassifier

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
# LOAD DATASET
# =========================================================
@st.cache_data
def load_dataset():
    return pd.read_csv("final_classification_dataset.csv")

df = load_dataset()

# =========================================================
# TRAIN MODEL FROM LATEST DATASET
# =========================================================
@st.cache_resource
def train_model_from_latest_dataset(dataframe):
    df_model = dataframe.copy()

    target_map = {
        "Low": 0,
        "Moderate": 1,
        "High": 2
    }

    X = df_model.drop(columns=["productivity_level"])
    y = df_model["productivity_level"].map(target_map)

    X_encoded = pd.get_dummies(
        X,
        columns=["quarter", "department", "day"],
        drop_first=False
    )

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=5,
        gamma=1,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=5,
        reg_alpha=0,
        random_state=42,
        eval_metric="mlogloss"
    )

    model.fit(X_encoded, y)

    return model, X_encoded.columns.tolist()

model, model_columns = train_model_from_latest_dataset(df)

# =========================================================
# SESSION STATE
# =========================================================
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# =========================================================
# DATA-DRIVEN OPTIONS
# =========================================================
quarter_options = sorted(df["quarter"].dropna().unique().tolist())
department_options = sorted(df["department"].dropna().unique().tolist())
day_options = sorted(df["day"].dropna().unique().tolist())
style_change_options = sorted(df["no_of_style_change"].dropna().astype(int).unique().tolist())

# =========================================================
# DATA-DRIVEN DEFAULTS AND LIMITS
# =========================================================
smv_min, smv_max = float(df["smv"].min()), float(df["smv"].max())
wip_min, wip_max = int(df["wip"].min()), int(df["wip"].max())
over_time_min, over_time_max = int(df["over_time"].min()), int(df["over_time"].max())
incentive_min, incentive_max = int(df["incentive"].min()), int(df["incentive"].max())
idle_time_min, idle_time_max = int(df["idle_time"].min()), int(df["idle_time"].max())
idle_men_min, idle_men_max = int(df["idle_men"].min()), int(df["idle_men"].max())
workers_min, workers_max = int(df["no_of_workers"].min()), int(df["no_of_workers"].max())

default_quarter = df["quarter"].mode()[0]
default_department = df["department"].mode()[0]
default_day = df["day"].mode()[0]
default_style_change = int(df["no_of_style_change"].mode()[0])

default_smv = float(round(df["smv"].median(), 2))
default_wip = int(df["wip"].median())
default_over_time = int(df["over_time"].median())
default_incentive = int(df["incentive"].median())
default_idle_time = int(df["idle_time"].median())
default_idle_men = int(df["idle_men"].median())
default_workers = int(df["no_of_workers"].median())

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def build_input_dataframe(
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
    no_of_workers,
    model_columns
):
    raw_input = pd.DataFrame([{
        "quarter": quarter,
        "department": department,
        "day": day,
        "smv": float(smv),
        "wip": int(wip),
        "over_time": int(over_time),
        "incentive": int(incentive),
        "idle_time": int(idle_time),
        "idle_men": int(idle_men),
        "no_of_style_change": int(no_of_style_change),
        "no_of_workers": int(no_of_workers)
    }])

    encoded_input = pd.get_dummies(
        raw_input,
        columns=["quarter", "department", "day"],
        drop_first=False
    )

    encoded_input = encoded_input.reindex(columns=model_columns, fill_value=0)
    return encoded_input

def decode_prediction(pred):
    reverse_map = {
        0: "Low",
        1: "Moderate",
        2: "High"
    }
    return reverse_map.get(int(pred), str(pred))

# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="main-title">🧵 AI-Powered Garment Factory Productivity Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">A machine learning-based decision support prototype for predicting garment factory productivity using XGBoost.</div>',
    unsafe_allow_html=True
)
st.success("✅ System Status: Latest-dataset model loaded successfully and ready for prediction.")

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

The prediction is based on the **latest finalized dataset**.
""")

    st.markdown("---")
    st.subheader("📂 Inputs Used")
    st.write("""
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
    quarter = st.selectbox("Quarter", quarter_options, index=quarter_options.index(default_quarter))
    department = st.selectbox("Department", department_options, index=department_options.index(default_department))
    day = st.selectbox("Day of the Week", day_options, index=day_options.index(default_day))
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("⚙️ Production Factors")
    smv = st.number_input("SMV (Complexity)", min_value=smv_min, max_value=smv_max, value=default_smv, step=0.01, format="%.2f")
    wip = st.number_input("Work in Progress (WIP)", min_value=wip_min, max_value=wip_max, value=default_wip, step=1)
    no_of_style_change = st.selectbox("Number of Style Changes", style_change_options, index=style_change_options.index(default_style_change))
    no_of_workers = st.number_input("Number of Workers", min_value=workers_min, max_value=workers_max, value=default_workers, step=1)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("💰 Time & Efficiency Metrics")
    over_time = st.number_input("Overtime", min_value=over_time_min, max_value=over_time_max, value=default_over_time, step=1)
    incentive = st.number_input("Incentive Amount", min_value=incentive_min, max_value=incentive_max, value=default_incentive, step=1)
    idle_time = st.number_input("Idle Time (Mins)", min_value=idle_time_min, max_value=idle_time_max, value=default_idle_time, step=1)
    idle_men = st.number_input("Idle Workers Count", min_value=idle_men_min, max_value=idle_men_max, value=default_idle_men, step=1)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# INPUT MONITORING NOTES
# =========================================================
notes = []

if no_of_workers <= max(workers_min, 5):
    notes.append("Low worker count may reduce production capacity.")
if wip >= int(0.8 * wip_max):
    notes.append("High WIP may create congestion and slow production flow.")
if over_time >= int(0.8 * over_time_max):
    notes.append("High overtime may indicate pressure on operations.")
if idle_time >= max(1, int(0.4 * idle_time_max)):
    notes.append("High idle time may negatively affect productivity.")
if idle_men >= max(1, int(0.25 * idle_men_max)):
    notes.append("A high number of idle workers may indicate weak resource utilization.")
if int(no_of_style_change) > 0:
    notes.append("Style changes may interrupt production flow and reduce consistency.")
if incentive == 0:
    notes.append("Zero incentive may affect worker motivation in some production settings.")
if smv >= (0.8 * smv_max):
    notes.append("High SMV suggests a more complex production task.")

if notes:
    with st.expander("⚠️ Input Monitoring Notes"):
        for note in notes:
            st.write(f"- {note}")

# =========================================================
# PREDICTION
# =========================================================
st.divider()

if st.button("Generate Productivity Forecast", use_container_width=True):
    input_df = build_input_dataframe(
        quarter=quarter,
        department=department,
        day=day,
        smv=smv,
        wip=wip,
        over_time=over_time,
        incentive=incentive,
        idle_time=idle_time,
        idle_men=idle_men,
        no_of_style_change=no_of_style_change,
        no_of_workers=no_of_workers,
        model_columns=model_columns
    )

    pred_idx = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]

    result = decode_prediction(pred_idx)
    confidence = float(np.max(probs))

    st.markdown(f"## Predicted Tier: **{result}**")

    if result == "High":
        st.success(f"Confidence: {confidence:.2%} - Optimized production detected.")
        st.balloons()
    elif result == "Moderate":
        st.warning(f"Confidence: {confidence:.2%} - Operating within standard range.")
    else:
        st.error(f"Confidence: {confidence:.2%} - High risk of target shortfall.")

    history_row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Predicted Tier": result,
        "Confidence": f"{confidence:.2%}",
        "Quarter": quarter,
        "Department": department,
        "Day": day,
        "SMV": float(smv),
        "WIP": int(wip),
        "Overtime": int(over_time),
        "Incentive": int(incentive),
        "Idle Time": int(idle_time),
        "Idle Workers": int(idle_men),
        "Style Change": int(no_of_style_change),
        "Workers": int(no_of_workers)
    }
    st.session_state.prediction_history.append(history_row)

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
