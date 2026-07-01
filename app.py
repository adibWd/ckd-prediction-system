# ============================================================
# CKD Prediction System v4.0
# ML-DL Weighted Ensemble (XGBoost + BiGRU-Attention)
# Author : Mominul Islam
# ============================================================

import os
import warnings

# -----------------------------
# Environment Settings (Mac M-Series)
# -----------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

warnings.filterwarnings("ignore")

# ============================================================
# Libraries
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model, Model

# ============================================================
# Streamlit Configuration
# ============================================================

st.set_page_config(
    page_title="CKD Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Disable GPU (Mac)
# ============================================================

try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

# ============================================================
# Custom Clinical Attention Layer
# (must match the architecture used during training exactly)
# ============================================================


class ClinicalAttentionLayerFixed(Layer):

    def build(self, input_shape):
        self.attention_weight = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.attention_bias = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        score = tf.matmul(inputs, self.attention_weight)
        score = score + self.attention_bias
        attention = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(inputs * attention, axis=1)
        return context, attention


# ============================================================
# Final model schema (LOCKED — must match training exactly)
# ============================================================
#
# These are the 12 features the saved scaler.pkl, xgboost_model.pkl
# and ckd_bigru_attention_final.keras were actually trained on
# (the "Improved Exp-3" feature set from the thesis notebook).
# This is NOT the same feature list that was in the old UI —
# that mismatch is the real reason BiGRU predictions looked broken.

FEATURE_COLUMNS = [
    "age",
    "bmi",
    "weight_kg",
    "height_cm",
    "bp_systolic",
    "bp_diastolic",
    "albumin_serum",
    "phosphorus",
    "bicarbonate",
    "calcium",
    "uric_acid",
    "urine_albumin",
]

# Locked final ensemble configuration (from the thesis notebook)
WEIGHT_XGB = 0.60
WEIGHT_BIGRU = 0.40
DECISION_THRESHOLD = 0.48

# ============================================================
# Load AI Models
# ============================================================


@st.cache_resource(show_spinner=False)
def load_models():

    scaler = joblib.load("models/scaler.pkl")

    xgb_model = joblib.load("models/xgboost_model.pkl")

    with tf.keras.utils.custom_object_scope(
        {"ClinicalAttentionLayerFixed": ClinicalAttentionLayerFixed}
    ):
        bigru_model = load_model(
            "models/ckd_bigru_attention_final.keras",
            compile=False,
        )

    # Build a small sub-model that exposes the attention weights
    # for explainability (which clinical features the BiGRU focused on)
    attention_layer = bigru_model.get_layer("clinical_attention")
    attention_extractor = Model(
        inputs=bigru_model.input,
        outputs=attention_layer.output[1],
    )

    # Warm-up call: forces TensorFlow to trace the computation graph now
    # (during app startup, under the loading spinner) instead of during
    # the first Predict click, which is what was causing the long hang.
    dummy_input = np.zeros((1, len(FEATURE_COLUMNS), 1), dtype="float32")
    bigru_model(dummy_input, training=False)
    attention_extractor(dummy_input, training=False)

    return scaler, xgb_model, bigru_model, attention_extractor


try:
    with st.spinner("Loading AI Models..."):
        scaler, xgb_model, bigru_model, attention_extractor = load_models()
    MODEL_READY = True

except Exception as e:
    MODEL_READY = False
    st.error("Model Loading Failed")
    st.exception(e)
    st.stop()

# ============================================================
# Header
# ============================================================

st.title("🏥 Chronic Kidney Disease Prediction System")

st.markdown(
    """
### AI-Based ML-DL Weighted Ensemble

This system combines

- 🟢 **XGBoost** (classical ML, weight = 0.60)
- 🔵 **BiGRU with Clinical Attention** (deep learning, weight = 0.40)
- 🟣 **Weighted Ensemble** → final decision threshold = 0.48

to estimate CKD risk from 12 clinical variables.
"""
)

st.success("✅ All AI Models Loaded Successfully")

st.divider()

# ============================================================
# Dashboard
# ============================================================

col1, col2, col3 = st.columns(3)
col1.metric("Machine Learning", "XGBoost")
col2.metric("Deep Learning", "BiGRU-Attention")
col3.metric("Clinical Features", "12")

st.divider()

# ============================================================
# Sidebar : Patient Information
# (inputs match the exact 12 features the models were trained on)
# ============================================================

st.sidebar.header("🩺 Patient Clinical Information")

age = st.sidebar.number_input("Age (years)", min_value=1, max_value=120, value=45)

bmi = st.sidebar.number_input(
    "Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=27.0
)

weight_kg = st.sidebar.number_input(
    "Weight (kg)", min_value=20.0, max_value=250.0, value=74.0
)

height_cm = st.sidebar.number_input(
    "Height (cm)", min_value=100.0, max_value=220.0, value=163.0
)

bp_systolic = st.sidebar.number_input(
    "Systolic Blood Pressure (mmHg)", min_value=60, max_value=240, value=118
)

bp_diastolic = st.sidebar.number_input(
    "Diastolic Blood Pressure (mmHg)", min_value=40, max_value=140, value=72
)

albumin_serum = st.sidebar.number_input(
    "Serum Albumin (g/dL)", min_value=1.0, max_value=6.0, value=4.1
)

phosphorus = st.sidebar.number_input(
    "Phosphorus (mg/dL)", min_value=1.0, max_value=12.0, value=3.6
)

bicarbonate = st.sidebar.number_input(
    "Bicarbonate (mEq/L)", min_value=5.0, max_value=40.0, value=24.6
)

calcium = st.sidebar.number_input(
    "Calcium (mg/dL)", min_value=5.0, max_value=14.0, value=9.4
)

uric_acid = st.sidebar.number_input(
    "Uric Acid (mg/dL)", min_value=1.0, max_value=15.0, value=5.0
)

urine_albumin = st.sidebar.number_input(
    "Urine Albumin (mg/L or ACR)", min_value=0.0, max_value=500.0, value=23.0
)

st.sidebar.divider()

predict_btn = st.sidebar.button(
    "🚀 Predict CKD", width="stretch", type="primary"
)

# ============================================================
# Feature Vector (column order MUST match training order)
# ============================================================

input_df = pd.DataFrame(
    [[
        age,
        bmi,
        weight_kg,
        height_cm,
        bp_systolic,
        bp_diastolic,
        albumin_serum,
        phosphorus,
        bicarbonate,
        calcium,
        uric_acid,
        urine_albumin,
    ]],
    columns=FEATURE_COLUMNS,
)

# ============================================================
# Main Dashboard — Patient Summary
# ============================================================

st.subheader("📋 Patient Summary")

left, right = st.columns(2)

with left:
    st.write(f"**Age:** {age}")
    st.write(f"**BMI:** {bmi}")
    st.write(f"**Weight:** {weight_kg} kg")
    st.write(f"**Height:** {height_cm} cm")
    st.write(f"**Systolic BP:** {bp_systolic} mmHg")
    st.write(f"**Diastolic BP:** {bp_diastolic} mmHg")

with right:
    st.write(f"**Serum Albumin:** {albumin_serum} g/dL")
    st.write(f"**Phosphorus:** {phosphorus} mg/dL")
    st.write(f"**Bicarbonate:** {bicarbonate} mEq/L")
    st.write(f"**Calcium:** {calcium} mg/dL")
    st.write(f"**Uric Acid:** {uric_acid} mg/dL")
    st.write(f"**Urine Albumin:** {urine_albumin}")

st.divider()

st.subheader("📊 Clinical Feature Table")

st.dataframe(input_df, width="stretch", hide_index=True)

# ============================================================
# Prediction Pipeline : XGBoost + BiGRU + Weighted Ensemble
# ============================================================

if predict_btn:

    st.divider()
    st.subheader("⚡ Prediction Progress")

    progress = st.progress(0)
    status = st.empty()

    try:
        # --------------------------------------------
        # Step 1 : XGBoost prediction (raw / unscaled features)
        # --------------------------------------------
        status.info("Step 1 / 3 : Running XGBoost Model...")
        progress.progress(33)

        xgb_probability = float(xgb_model.predict_proba(input_df)[0][1])

        # --------------------------------------------
        # Step 2 : Scale features + run BiGRU-Attention
        # --------------------------------------------
        status.info("Step 2 / 3 : Running BiGRU-Attention Model...")
        progress.progress(66)

        scaled_data = scaler.transform(input_df)
        reshaped_data = scaled_data.reshape(1, scaled_data.shape[1], 1).astype("float32")

        bigru_probability = float(bigru_model(reshaped_data, training=False).numpy()[0][0])
        attention_weights = attention_extractor(reshaped_data, training=False).numpy().ravel()

        # --------------------------------------------
        # Step 3 : Weighted ensemble
        # --------------------------------------------
        status.info("Step 3 / 3 : Combining into Final Ensemble...")
        progress.progress(100)

        ensemble_probability = (WEIGHT_XGB * xgb_probability) + (
            WEIGHT_BIGRU * bigru_probability
        )

        status.success("✅ Prediction Completed")

        # --------------------------------------------
        # Results
        # --------------------------------------------
        st.divider()
        st.subheader("🧠 Model Results")

        c1, c2, c3 = st.columns(3)

        c1.metric("XGBoost Probability", f"{xgb_probability*100:.2f}%")
        c2.metric("BiGRU-Attention Probability", f"{bigru_probability*100:.2f}%")
        c3.metric("Ensemble Probability", f"{ensemble_probability*100:.2f}%")

        st.divider()
        st.subheader("🎯 Final Ensemble Decision")

        prediction_label = (
            "CKD Detected" if ensemble_probability >= DECISION_THRESHOLD else "Healthy"
        )

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediction", prediction_label)
        with col2:
            st.metric("Decision Threshold", f"{DECISION_THRESHOLD:.2f}")

        st.progress(float(min(max(ensemble_probability, 0.0), 1.0)))

        if ensemble_probability >= DECISION_THRESHOLD:
            st.error(f"🚨 High CKD Risk ({ensemble_probability*100:.2f}%)")
        else:
            st.success(f"🟢 Low CKD Risk ({ensemble_probability*100:.2f}%)")

        # --------------------------------------------
        # Explainability : Clinical Attention Weights
        # --------------------------------------------
        st.divider()
        st.subheader("🔬 BiGRU Clinical Attention (Explainability)")
        st.caption(
            "Shows which clinical features the BiGRU-Attention model focused on most "
            "for this specific patient."
        )

        attention_df = pd.DataFrame(
            {
                "Feature": FEATURE_COLUMNS,
                "Attention Weight": attention_weights,
            }
        ).sort_values("Attention Weight", ascending=False)

        st.bar_chart(attention_df.set_index("Feature"))

        # --------------------------------------------
        # Debug Section
        # --------------------------------------------
        with st.expander("🔍 Debug Information"):
            st.write("Raw Feature Shape")
            st.code(str(input_df.shape))

            st.write("Scaled Feature Shape (BiGRU input before reshape)")
            st.code(str(scaled_data.shape))

            st.write("BiGRU Reshaped Input")
            st.code(str(reshaped_data.shape))

            st.write("XGBoost Probability")
            st.code(str(xgb_probability))

            st.write("BiGRU Probability")
            st.code(str(bigru_probability))

            st.write("Ensemble Weights")
            st.code(f"XGBoost = {WEIGHT_XGB}, BiGRU = {WEIGHT_BIGRU}")

            st.write("Scaled Features")
            st.dataframe(
                pd.DataFrame(scaled_data, columns=FEATURE_COLUMNS),
                width="stretch",
            )

    except Exception as e:
        st.error("❌ Prediction Failed")
        st.exception(e)
