import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


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
    .pill {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        background: #eef2ff;
        border: 1px solid #dbe4ff;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# FILE HELPERS
# =========================================================
def find_existing_file(candidates):
    for name in candidates:
        if Path(name).exists():
            return name
    return None

MODEL_CANDIDATES = [
    "garment_xgb_model.pkl",
    "garment_xgb_model_latest.pkl",
    "garment_xgb_model_latest (1).pkl",
]

COLUMN_CANDIDATES = [
    "xgb_model_columns.pkl",
    "xgb_model_columns_latest.pkl",
    "xgb_model_columns_latest (1).pkl",
]

DATASET_CANDIDATES = [
    "final_classification_dataset.csv",
]

# =========================================================
# LOAD ASSETS
# =========================================================
@st.cache_resource
def load_model_assets():
    model_path = find_existing_file(MODEL_CANDIDATES)
    columns_path = find_existing_file(COLUMN_CANDIDATES)

    if model_path is None or columns_path is None:
        missing = []
        if model_path is None:
            missing.append("model .pkl file")
        if columns_path is None:
            missing.append("model columns .pkl file")
        raise FileNotFoundError(f"Missing required file(s): {', '.join(missing)}")

    model = joblib.load(model_path)
    model_columns = joblib.load(columns_path)
    return model, model_columns, model_path, columns_path

@st.cache_data
def load_dataset():
    dataset_path = find_existing_file(DATASET_CANDIDATES)
    if dataset_path is None:
        raise FileNotFoundError("Missing dataset file: final_classification_dataset.csv")
    return pd.read_csv(dataset_path), dataset_path

model, model_columns, model_path, columns_path = load_model_assets()
df, dataset_path = load_dataset()

# =========================================================
# DATA PREPARATION
# =========================================================
def ordered_existing(values, preferred_order):
    values_set = set(values)
    ordered = [x for x in preferred_order if x in values_set]
    remaining = [x for x in values if x not in ordered]
    return ordered + sorted(remaining)

quarter_options = ordered_existing(
    df["quarter"].dropna().unique().tolist(),
    ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"]
)

department_options = ordered_existing(
    df["department"].dropna().unique().tolist(),
    ["sewing", "finished"]
)

day_options = ordered_existing(
    df["day"].dropna().unique().tolist(),
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"]
)

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
            return True
    return False

def normalize_prediction(pred):
    class_map = {0: "Low", 1: "Moderate", 2: "High"}
    if isinstance(pred, str):
        return pred
    try:
        return class_map.get(int(pred), str(pred))
    except Exception:
        return str(pred)

def label_from_model_class(raw_class):
    class_map = {0: "Low", 1: "Moderate", 2: "High"}
    if isinstance(raw_class, str):
        return raw_class
    try:
        return class_map.get(int(raw_class), str(raw_class))
    except Exception:
        return str(raw_class)

def normalize_dataset_label(value):
    if pd.isna(value):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v == "low":
            return "Low"
        elif v == "moderate":
            return "Moderate"
        elif v == "high":
            return "High"
        return value
    class_map = {0: "Low", 1: "Moderate", 2: "High"}
    try:
        return class_map.get(int(value), str(value))
    except Exception:
        return str(value)

def get_result_message(result):
    if result == "High":
        return "success", "The current input pattern suggests strong production performance."
    elif result == "Moderate":
        return "warning", "The current input pattern suggests moderate and relatively stable production performance."
    else:
        return "error", "The current input pattern suggests a risk of lower productivity."

def get_recommendations(result):
    recs = []

    if result == "Low":
        recs.append("Review the current production setup and identify operational factors that may be lowering efficiency.")
        recs.append("Improve workflow planning and reduce interruptions where possible.")
        recs.append("Monitor labour utilization, overtime, and process consistency more closely.")
    elif result == "Moderate":
        recs.append("The production line is operating within a normal range but still has room for improvement.")
        recs.append("Better workload planning and smoother operations may help raise productivity to the high tier.")
        recs.append("Continue monitoring process consistency to maintain stable performance.")
    else:
        recs.append("The current production setup appears efficient and well balanced.")
        recs.append("This input combination can be used as a benchmark for future planning.")
        recs.append("Maintain consistent resource allocation and stable workflow conditions to sustain high productivity.")

    return recs

def get_reference_class_snapshot(df_input):
    temp_df = df_input.copy()
    temp_df["productivity_level"] = temp_df["productivity_level"].apply(normalize_dataset_label)

    summary = (
        temp_df.groupby("productivity_level", dropna=False)
        .agg({
            "smv": "median",
            "wip": "median",
            "over_time": "median",
            "incentive": "median",
            "idle_time": "median",
            "idle_men": "median",
            "no_of_workers": "median",
        })
        .reset_index()
    )

    desired_order = ["Low", "Moderate", "High"]
    summary["sort_order"] = summary["productivity_level"].apply(
        lambda x: desired_order.index(x) if x in desired_order else 999
    )
    summary = summary.sort_values("sort_order").drop(columns="sort_order")
    return summary

# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="main-title">🧵 AI-Powered Garment Factory Productivity Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">A machine learning-based decision support prototype for predicting garment factory productivity using XGBoost.</div>',
    unsafe_allow_html=True
)

st.success("✅ System Status: Model and dataset loaded successfully.")

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
    st.subheader("📂 Active Files")
    st.caption(f"Model: {model_path}")
    st.caption(f"Columns: {columns_path}")
    st.caption(f"Dataset: {dataset_path}")

    st.markdown("---")
    st.subheader("📂 Dataset-Aligned Inputs")
    st.write("""
This app uses these finalized dataset fields:
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
flagged_fields = []

if idle_time > df["idle_time"].quantile(0.90):
    flagged_fields.append("Idle Time")
if idle_men > df["idle_men"].quantile(0.90):
    flagged_fields.append("Idle Workers")
if over_time > df["over_time"].quantile(0.90):
    flagged_fields.append("Overtime")
if wip > df["wip"].quantile(0.90):
    flagged_fields.append("WIP")

if flagged_fields:
    with st.expander("⚠️ Input Monitoring Notes"):
        st.write("One or more input values are relatively unusual compared with the dataset records.")
        st.write("Flagged inputs: " + ", ".join(flagged_fields))

# =========================================================
# PREDICTION
# =========================================================
st.divider()

if st.button("Generate Productivity Forecast", use_container_width=True):
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)

    numeric_fields = {
        "smv": smv,
        "wip": wip,
        "over_time": over_time,
        "incentive": incentive,
        "idle_time": idle_time,
        "idle_men": idle_men,
        "no_of_workers": no_of_workers,
    }

    for col, val in numeric_fields.items():
        if col in input_df.columns:
            input_df[col] = val

    if "no_of_style_change" in input_df.columns:
        input_df["no_of_style_change"] = no_of_style_change
    else:
        set_dummy_value(input_df, "no_of_style_change", no_of_style_change)

    set_dummy_value(input_df, "quarter", quarter)
    set_dummy_value(input_df, "department", department)
    set_dummy_value(input_df, "day", day)

    input_df = input_df[model_columns]

    raw_pred = model.predict(input_df)[0]
    result = normalize_prediction(raw_pred)

    probs = None
    prob_df = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(input_df)[0]
            class_labels = getattr(model, "classes_", [0, 1, 2])
            readable_labels = [label_from_model_class(x) for x in class_labels]

            prob_df = pd.DataFrame({
                "Productivity Level": readable_labels,
                "Probability": probs
            })

            class_order = ["Low", "Moderate", "High"]
            prob_df["sort_order"] = prob_df["Productivity Level"].apply(
                lambda x: class_order.index(x) if x in class_order else 999
            )
            prob_df = prob_df.sort_values("sort_order").drop(columns="sort_order").reset_index(drop=True)
        except Exception:
            probs = None
            prob_df = None

    status_type, status_msg = get_result_message(result)

    st.markdown("## 📊 Prediction Results")

    if probs is not None:
        confidence = float(np.max(probs))
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Productivity", result)
        c2.metric("Confidence Score", f"{confidence:.2%}")
        c3.metric("Workers", int(no_of_workers))

        st.caption(
            "Confidence score refers to the highest predicted probability produced by the model for the selected class. "
            "It shows how strongly the model prefers that class compared with the other classes."
        )

        if confidence < 0.55:
            st.warning("Prediction confidence is relatively low. Interpret this result carefully.")
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

    if result == "High":
        download_df = pd.DataFrame({
            "Feature": [
                "Quarter", "Department", "Day", "SMV", "WIP", "Overtime",
                "Incentive", "Idle Time", "Idle Workers", "Style Changes",
                "Number of Workers", "Predicted Productivity"
            ],
            "Value": [
                quarter, department, day, smv, wip, over_time,
                incentive, idle_time, idle_men, no_of_style_change,
                no_of_workers, result
            ]
        })

        csv_data = download_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download High Productivity Result",
            data=csv_data,
            file_name="high_productivity_result.csv",
            mime="text/csv"
        )
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📌 Input Summary",
        "📈 Confidence Breakdown",
        "💡 Recommendations",
        "🧾 Encoded Model Input",
        "📚 Dataset Reference"
    ])

    with tab1:
        summary_df = pd.DataFrame({
            "Feature": [
                "Quarter", "Department", "Day", "SMV", "WIP", "Overtime",
                "Incentive", "Idle Time", "Idle Workers", "Style Changes", "Number of Workers"
            ],
            "Value": [
                quarter, department, day, smv, wip, over_time,
                incentive, idle_time, idle_men, no_of_style_change, no_of_workers
            ]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
    with tab2:
        if prob_df is not None:
            st.caption("This chart shows the predicted probability for each productivity class.")
            
            ordered_prob_df = prob_df.set_index("Productivity Level").reindex(["Low", "Moderate", "High"])
            st.bar_chart(ordered_prob_df)
            
            display_df = ordered_prob_df.reset_index().copy()
            display_df["Probability"] = display_df["Probability"].map(lambda x: f"{x:.2%}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("Probability output is not available for the current loaded model.")
        
    with tab3:
        recommendations = get_recommendations(result)
        for i, rec in enumerate(recommendations, start=1):
            st.write(f"**{i}.** {rec}")

    with tab4:
        st.caption(
            "This table shows the final encoded values that are actually sent into the trained model. "
            "Categorical selections are converted into numeric model features such as dummy variables."
        )
        encoded_preview = input_df.T.reset_index()
        encoded_preview.columns = ["Model Feature", "Value"]
        encoded_preview = encoded_preview[encoded_preview["Value"] != 0]
        st.dataframe(encoded_preview, use_container_width=True, hide_index=True)

    with tab5:
        ref_df = get_reference_class_snapshot(df)
        st.caption("Median feature values by productivity class in the dataset.")
        st.dataframe(ref_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown(
    '<div class="small-note">Prototype purpose: To support production planning, labor monitoring, and productivity forecasting in garment factory operations.</div>',
    unsafe_allow_html=True
)
