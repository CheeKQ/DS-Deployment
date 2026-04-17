import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="Garment Productivity Predictor", layout="wide")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    model = joblib.load("garment_xgb_model.pkl")
    model_columns = joblib.load("xgb_model_columns.pkl")
    return model, model_columns

@st.cache_data
def load_dataset():
    return pd.read_csv("final_classification_dataset.csv")

model, model_columns = load_assets()
df = load_dataset()

# --- SESSION STATE FOR HISTORY ---
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# --- DATASET-BASED OPTIONS ---
day_options = sorted(df["day"].dropna().unique().tolist())
quarter_options = sorted(df["quarter"].dropna().unique().tolist())
department_options = sorted(df["department"].dropna().unique().tolist())
style_change_options = sorted(df["no_of_style_change"].dropna().astype(int).unique().tolist())

# --- DATASET-BASED LIMITS ---
smv_min = float(df["smv"].min())
smv_max = float(df["smv"].max())

wip_min = int(df["wip"].min())
wip_max = int(df["wip"].max())

over_time_min = int(df["over_time"].min())
over_time_max = int(df["over_time"].max())

incentive_min = int(df["incentive"].min())
incentive_max = int(df["incentive"].max())

idle_time_min = int(df["idle_time"].min())
idle_time_max = int(df["idle_time"].max())

idle_men_min = int(df["idle_men"].min())
idle_men_max = int(df["idle_men"].max())

workers_min = int(df["no_of_workers"].min())
workers_max = int(df["no_of_workers"].max())

# --- DEFAULT VALUES FROM DATASET MEDIAN / MODE ---
default_day = df["day"].mode()[0]
default_quarter = df["quarter"].mode()[0]
default_department = df["department"].mode()[0]
default_style_change = int(df["no_of_style_change"].mode()[0])

default_smv = float(round(df["smv"].median(), 2))
default_wip = int(df["wip"].median())
default_over_time = int(df["over_time"].median())
default_incentive = int(df["incentive"].median())
default_idle_time = int(df["idle_time"].median())
default_idle_men = int(df["idle_men"].median())
default_workers = int(df["no_of_workers"].median())

# --- UI DESIGN ---
st.title("🧵 Garment Factory Productivity Predictor")
st.info("""**Model Info:** Currently using a **Tuned XGBoost Classifier**. 
This model evaluates production conditions and predicts garment factory productivity level.""")

# We use this to track if ANY input is currently invalid
form_is_invalid = False

# --- INPUT COLUMNS ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📅 Time & Place")
    day = st.selectbox("Day of the Week", day_options, index=day_options.index(default_day))
    quarter = st.selectbox("Quarter", quarter_options, index=quarter_options.index(default_quarter))
    dept = st.selectbox("Department", department_options, index=department_options.index(default_department))

with col2:
    st.subheader("⚙️ Resource Allocation")

    wip = st.number_input(
        "Work in Progress (WIP)",
        min_value=wip_min,
        max_value=wip_max,
        value=default_wip,
        step=1
    )

    workers = st.number_input(
        "Number of Workers",
        min_value=workers_min,
        max_value=workers_max,
        value=default_workers,
        step=1
    )

    style_change = st.selectbox(
        "Number of Style Changes",
        style_change_options,
        index=style_change_options.index(default_style_change)
    )

    smv = st.number_input(
        "SMV (Complexity)",
        min_value=smv_min,
        max_value=smv_max,
        value=default_smv,
        step=0.01,
        format="%.2f"
    )

with col3:
    st.subheader("💰 Incentives & Metrics")

    incentive = st.number_input(
        "Incentive Amount",
        min_value=incentive_min,
        max_value=incentive_max,
        value=default_incentive,
        step=1
    )

    over_time = st.number_input(
        "Overtime",
        min_value=over_time_min,
        max_value=over_time_max,
        value=default_over_time,
        step=1
    )

    idle_time = st.number_input(
        "Idle Time (Mins)",
        min_value=idle_time_min,
        max_value=idle_time_max,
        value=default_idle_time,
        step=1
    )

    idle_men = st.number_input(
        "Idle Workers Count",
        min_value=idle_men_min,
        max_value=idle_men_max,
        value=default_idle_men,
        step=1
    )

# --- INPUT MONITORING NOTES ---
notes = []

if workers <= max(workers_min, 5):
    notes.append("Low worker count may reduce production capacity.")
if wip >= int(0.8 * wip_max):
    notes.append("High WIP may create congestion and slow production flow.")
if over_time >= int(0.8 * over_time_max):
    notes.append("High overtime may indicate pressure on operations.")
if idle_time >= max(1, int(0.4 * idle_time_max)):
    notes.append("High idle time may negatively affect productivity.")
if idle_men >= max(1, int(0.25 * idle_men_max)):
    notes.append("A high number of idle workers may indicate weak resource utilization.")
if int(style_change) > 0:
    notes.append("Style changes may interrupt production flow and reduce consistency.")
if incentive == 0:
    notes.append("Zero incentive may affect worker motivation in some production settings.")
if smv >= (0.8 * smv_max):
    notes.append("High SMV suggests a more complex production task, which may affect output.")

if notes:
    with st.expander("⚠️ Input Monitoring Notes"):
        for note in notes:
            st.write(f"- {note}")

# --- PREDICTION LOGIC ---
st.divider()

def set_dummy(category, value):
    candidates = [
        f"{category}_{value}",
        f"{category}_{str(value).lower()}",
        f"{category}_{str(value).upper()}",
        f"{category}_{str(value).capitalize()}",
    ]
    for col_name in candidates:
        if col_name in model_columns:
            input_df[col_name] = 1
            return

if form_is_invalid:
    st.warning("Please correct the errors above to enable the prediction.")
    st.button("Generate Productivity Forecast", disabled=True)
else:
    if st.button("Generate Productivity Forecast", use_container_width=True):
        # Initialize DataFrame
        input_df = pd.DataFrame(0, index=[0], columns=model_columns)

        # Numeric mapping
        numeric_map = {
            "smv": float(smv),
            "wip": float(wip),
            "over_time": int(over_time),
            "incentive": int(incentive),
            "idle_time": float(idle_time),
            "idle_men": int(idle_men),
            "no_of_style_change": int(style_change),
            "no_of_workers": float(workers),
        }

        for col, val in numeric_map.items():
            if col in model_columns:
                input_df[col] = val

        # Encoding for categorical columns if needed
        set_dummy("quarter", quarter)
        set_dummy("department", dept)
        set_dummy("day", day)
        set_dummy("no_of_style_change", style_change)

        # Align & Predict
        input_df = input_df[model_columns]
        raw_prediction = model.predict(input_df)[0]

        # Handle both numeric and text labels
        if isinstance(raw_prediction, str):
            result = raw_prediction
        else:
            labels = ["Low", "Moderate", "High"]
            result = labels[int(raw_prediction)]

        # Predict probabilities if available
        probs = None
        confidence = None
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(input_df)[0]
                confidence = float(np.max(probs))
            except Exception:
                probs = None
                confidence = None

        st.markdown(f"## Predicted Tier: **{result}**")
        if confidence is not None:
            if result == "High":
                st.success(f"Confidence: {confidence:.2%} - Optimized production detected.")
                st.balloons()
            elif result == "Moderate":
                st.warning(f"Confidence: {confidence:.2%} - Operating within standard range.")
            else:
                st.error(f"Confidence: {confidence:.2%} - High risk of target shortfall.")
        else:
            if result == "High":
                st.success("Optimized production detected.")
                st.balloons()
            elif result == "Moderate":
                st.warning("Operating within standard range.")
            else:
                st.error("High risk of target shortfall.")

        # --- STORE CURRENT RESULT IN HISTORY ---
        history_row = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Predicted Tier": result,
            "Confidence": f"{confidence:.2%}" if confidence is not None else "N/A",
            "Day": day,
            "Quarter": quarter,
            "Department": dept,
            "WIP": float(wip),
            "Workers": float(workers),
            "Style Change": int(style_change),
            "SMV": float(smv),
            "Incentive": int(incentive),
            "Overtime": int(over_time),
            "Idle Time": float(idle_time),
            "Idle Workers": int(idle_men)
        }
        st.session_state.prediction_history.append(history_row)

        # --- EXPORT CURRENT RESULT ---
        st.subheader("📥 Export Current Result")
        current_result_df = pd.DataFrame([history_row])
        current_csv = current_result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Current Prediction as CSV",
            data=current_csv,
            file_name="current_prediction_result.csv",
            mime="text/csv"
        )

# --- PREDICTION HISTORY ---
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
