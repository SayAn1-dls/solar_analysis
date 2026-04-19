"""
Streamlit Dashboard for the Solar Power Forecasting project.

Provides interactive tabs for:
  - Single & batch prediction
  - Exploratory data analysis
  - Model evaluation & explainability
  - Future hourly forecasting
  - Training logs & data export

Run:
  streamlit run app/streamlit_app.py
"""

import sys
import os
import json
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Try to import local modules with error handling for deployment
try:
    from src.preprocessing.preprocessing import encode_features, FEATURES, TARGET
    from src.evaluation.metrics import compute_mape
    from src.utils.helpers import style_plot, concept_note, C_RF, C_ACTUAL, C_ACCENT, C_GOLD
    from src.agentic_rag.workflow import run_agentic_grid_optimizer
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# ══════════════════════════════════════
# PAGE CONFIG & STYLING
# ══════════════════════════════════════
st.set_page_config(
    page_title="Solar Power Forecasting — ML Project",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1E88E5 0%, #43A047 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        padding: 1rem 0 0.3rem 0;
        letter-spacing: -1px;
    }
    .sub-header {
        text-align: center;
        color: #9CA3AF;
        font-size: 1.1rem;
        font-weight: 400;
        margin-top: -8px;
        margin-bottom: 30px;
    }

    .concept-box {
        background: rgba(30, 136, 229, 0.08);
        border-left: 4px solid #1E88E5;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 12px 0 20px 0;
        font-size: 0.92rem;
        color: #D1D5DB;
        line-height: 1.65;
    }
    .concept-box strong { color: #60A5FA; }

    div[data-testid="stMetric"] {
        background: rgba(17, 24, 39, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 18px 22px;
        box-shadow: 0 8px 25px -8px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px -10px rgba(30, 136, 229, 0.25);
        border-color: rgba(30, 136, 229, 0.4);
    }
    div[data-testid="stMetric"] label {
        color: #9CA3AF !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #F9FAFB !important;
        font-weight: 700 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 12px 22px;
        font-weight: 600;
        font-size: 0.95rem;
        color: #9CA3AF;
        border-bottom: 3px solid transparent;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #E5E7EB;
        background-color: rgba(255, 255, 255, 0.04);
    }
    .stTabs [aria-selected="true"] {
        color: #1E88E5 !important;
        border-bottom: 3px solid #1E88E5 !important;
        background-color: rgba(30, 136, 229, 0.06) !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        transition: all 0.25s ease;
        box-shadow: 0 4px 14px 0 rgba(30, 136, 229, 0.35);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(30, 136, 229, 0.45);
    }

    .section-divider {
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        margin: 35px 0 25px 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Solar Power Forecasting</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">'
    'Supervised Learning Project — Decision Tree Regression with Real-World Solar Data'
    '</p>',
    unsafe_allow_html=True,
)


# ══════════════════════════════════════
# LOAD MODEL + DATASET
# ══════════════════════════════════════
@st.cache_resource
def load_model():
    try:
        # Try multiple possible paths for model file
        import os
        model_paths = [
            "models/solar_model.pkl",
            os.path.join(os.getcwd(), "models/solar_model.pkl"),
            "/mount/src/solar_analysis/models/solar_model.pkl"
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                return joblib.load(path)
        
        # If no model found, show error with paths checked
        st.error("Model file not found. Please ensure models/solar_model.pkl exists.")
        st.write("Paths checked:")
        for path in model_paths:
            st.write(f"- {path}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()


@st.cache_data
def load_dataset():
    try:
        # Try multiple possible paths for dataset file
        import os
        data_paths = [
            "data/processed/solar_final.csv",
            os.path.join(os.getcwd(), "data/processed/solar_final.csv"),
            "/mount/src/solar_analysis/data/processed/solar_final.csv"
        ]
        
        for path in data_paths:
            if os.path.exists(path):
                return pd.read_csv(path)
        
        # If no dataset found, show error with paths checked
        st.error("Dataset file not found. Please ensure data/processed/solar_final.csv exists.")
        st.write("Paths checked:")
        for path in data_paths:
            st.write(f"- {path}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()


@st.cache_data
def load_training_log():
    for path in ["training_log.json", "models/training_log.json"]:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return None


df_full = load_dataset()
df_train = encode_features(df_full.copy())
training_log = load_training_log()


# ══════════════════════════════════════
# TIME-BASED TRAIN-TEST SPLIT
# ══════════════════════════════════════
df_sorted = df_full.sort_values("DATE_TIME")
split_idx = int(len(df_sorted) * 0.8)

# Encode after sorting to preserve DATE_TIME
df_sorted_encoded = encode_features(df_sorted.copy())
X_all = df_sorted_encoded[FEATURES]
y_all = df_sorted[TARGET]

X_train_ts = X_all.iloc[:split_idx]
X_test_ts = X_all.iloc[split_idx:]
y_train_ts = y_all.iloc[:split_idx]
y_test_ts = y_all.iloc[split_idx:]


# ══════════════════════════════════════
# TABS
# ══════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Predict",
    "Data Analysis",
    "Model Evaluation",
    "Forecast",
    "Logs & Export",
    "AI Grid Optimizer",
])


# ══════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════
with tab1:
    st.header("Supervised Regression — Predict Solar Power")

    concept_note(
        "<strong>Concept — Supervised Learning (Regression):</strong> "
        "The model learns a mapping from input features (weather, time) to a continuous target "
        "(DC Power in Watts). This is a <strong>regression</strong> task under supervised learning, "
        "where the model is trained on labelled historical data."
    )

    col_in1, col_in2 = st.columns(2)
    with col_in1:
        SOURCE_KEY = st.number_input("Inverter ID (encoded)", 0, key="pred_src")
        AMBIENT_TEMPERATURE = st.number_input("Ambient Temperature (°C)", value=28.0, key="pred_amb")
        MODULE_TEMPERATURE = st.number_input("Module Temperature (°C)", value=40.0, key="pred_mod")
    with col_in2:
        IRRADIATION = st.number_input("Irradiation (0.0 – 1.2)", value=0.5, key="pred_irr")
        hour = st.slider("Hour of Day", 0, 23, 12, key="pred_hour")
        month = st.slider("Month", 1, 12, 5, key="pred_month")

    if st.button("Predict Solar Power", type="primary", key="pred_btn"):

        data = pd.DataFrame([[SOURCE_KEY, AMBIENT_TEMPERATURE,
                               MODULE_TEMPERATURE, IRRADIATION,
                               hour, month]], columns=FEATURES)

        pred = max(0, load_model().predict(data)[0])

        st.metric("Predicted DC Power (RandomForest)", f"{pred:,.0f} W")

        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor("#0D1117")
        bar = ax.bar(["RandomForest"], [pred], color=[C_RF], alpha=0.9,
                     edgecolor="white", linewidth=1.2, width=0.4)
        ax.text(bar[0].get_x() + bar[0].get_width() / 2, bar[0].get_height() + 50,
                f'{pred:,.0f} W', ha='center', va='bottom', fontsize=11,
                fontweight="bold", color="#E5E7EB")
        style_plot(ax, "Predicted Solar Power Output", ylabel="Power (W)")
        plt.tight_layout()
        st.pyplot(fig)

    # ── Batch Prediction ──
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("Batch Prediction (CSV Upload)")

    uploaded_file = st.file_uploader("Upload weather CSV with required features", type=["csv"])

    if uploaded_file is not None:
        df_up = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:")
        st.dataframe(df_up.head(), use_container_width=True)

        if st.button("Run Batch Forecast", key="batch_btn"):
            missing = [f for f in FEATURES if f not in df_up.columns]
            if missing:
                st.error(f"❌ Missing required columns: {missing}")
                st.info("📋 Required columns for batch prediction:")
                for feature in FEATURES:
                    status = "✅" if feature in df_up.columns else "❌"
                    st.write(f"{status} {feature}")
                st.info("💡 **Tip**: Your CSV must contain these exact column names:")
                st.code(", ".join(FEATURES))
                st.info("📄 **Sample CSV format:**")
                sample_data = {
                    "SOURCE_KEY": ["A", "B"],
                    "AMBIENT_TEMPERATURE": [25.5, 30.2],
                    "MODULE_TEMPERATURE": [35.1, 40.5],
                    "IRRADIATION": [800, 950],
                    "hour": [12, 14],
                    "month": [6, 7]
                }
                st.dataframe(pd.DataFrame(sample_data))
            else:
                try:
                    df_pred = df_up.copy()
                    df_pred = encode_features(df_pred)

                    # Additional validation
                    if df_pred[FEATURES].isnull().any().any():
                        st.error("❌ Uploaded data contains missing values. Please clean your data.")
                        st.stop()

                    preds_batch = np.clip(load_model().predict(df_pred[FEATURES]), 0, None)
                    df_up["Prediction"] = preds_batch

                    st.success("✅ Forecast completed successfully!")
                    st.dataframe(df_up, use_container_width=True)

                    csv = df_up.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download Predictions", csv,
                                       "solar_predictions.csv", "text/csv")
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
                    st.info("Please check your data format and try again.")

    # ── Dataset Coverage ──
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("Training Data Coverage")

    months_present = sorted(df_full["month"].astype(int).unique().tolist())
    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    available = [month_names[m] for m in months_present]

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.write("**Seasonal coverage in training data:**", ", ".join(available))
        if month not in months_present:
            st.warning("Selected month is outside training data — higher error expected.")
        else:
            st.success("Model is trained on this seasonal pattern.")
    with col_c2:
        confidence = 0.9 if month in months_present else 0.6
        st.metric("Prediction Confidence", f"{confidence:.0%}")


# ══════════════════════════════════════════════════════
# TAB 2 — DATA ANALYSIS
# ══════════════════════════════════════════════════════
with tab2:
    st.header("Exploratory Data Analysis")

    concept_note(
        "<strong>Concept — Data Preprocessing:</strong> "
        "Before training, raw solar data undergoes feature engineering (extracting hour, month "
        "from timestamps) and encoding (converting categorical inverter IDs to numerical labels). "
        "These preprocessing steps are essential for building reliable regression models."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Power", f"{int(df_train['DC_POWER'].mean()):,} W")
    col2.metric("Peak Power", f"{int(df_train['DC_POWER'].max()):,} W")
    col3.metric("Std Deviation", f"{int(df_train['DC_POWER'].std()):,} W")
    col4.metric("Data Points", f"{len(df_train):,}")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("Seasonal Solar Trends")

    monthly = df_train.groupby("month")["DC_POWER"].mean()
    st.line_chart(monthly)
    st.caption("Training data covers May–June. "
               "Seasonal variation is captured through the 'month' feature.")

    concept_note(
        "<strong>Concept — Feature Engineering:</strong> "
        "The 'month' feature captures seasonal variation in solar irradiance. "
        "This is <strong>domain-driven feature engineering</strong> — using knowledge of the "
        "physical system to create informative features that improve model performance."
    )

    st.subheader("Average Daily Solar Curve")

    hourly = df_train.groupby("hour")["DC_POWER"].mean()

    fig_daily, ax_d = plt.subplots(figsize=(12, 4.5))
    fig_daily.patch.set_facecolor("#0D1117")
    ax_d.fill_between(hourly.index, hourly.values, alpha=0.25, color=C_ACCENT)
    ax_d.plot(hourly.index, hourly.values, color=C_ACCENT, linewidth=2.5,
              marker="o", markersize=5)
    style_plot(ax_d, "Average DC Power by Hour of Day", xlabel="Hour", ylabel="DC Power (W)")
    ax_d.set_xticks(range(0, 24))
    ax_d.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    plt.tight_layout()
    st.pyplot(fig_daily)

    st.info("Solar generation follows a bell curve aligned with sun elevation angle. "
            "Peak output occurs around noon — a nonlinear pattern that decision trees capture well.")

    st.subheader("Output Variability by Hour")

    variability = df_train.groupby("hour")["DC_POWER"].std()

    fig_var, ax_v = plt.subplots(figsize=(12, 4))
    fig_var.patch.set_facecolor("#0D1117")
    ax_v.bar(variability.index, variability.values, color=C_GOLD, alpha=0.8,
             edgecolor="white", linewidth=0.5)
    style_plot(ax_v, "Power Standard Deviation by Hour", xlabel="Hour", ylabel="Std Dev (W)")
    ax_v.set_xticks(range(0, 24))
    plt.tight_layout()
    st.pyplot(fig_var)

    concept_note(
        "<strong>Concept — Variance:</strong> "
        "High standard deviation at certain hours means the model faces greater "
        "<strong>variance</strong> in those regions. Understanding data variability "
        "helps diagnose the bias-variance tradeoff."
    )

    peak_hours = hourly.sort_values(ascending=False).head(3)
    st.success(f"**Peak generation hours:** {list(peak_hours.index)} "
               f"(avg power: {peak_hours.values[0]:,.0f} W)")


# ══════════════════════════════════════════════════════
# TAB 3 — MODEL EVALUATION
# ══════════════════════════════════════════════════════
with tab3:
    st.header("Model Evaluation")

    concept_note(
        "<strong>Concept — Regression Metrics:</strong> "
        "<strong>MAE</strong> measures average absolute error, "
        "<strong>RMSE</strong> penalizes large errors more heavily, "
        "<strong>R²</strong> indicates the proportion of variance explained (1.0 = perfect), and "
        "<strong>MAPE</strong> gives percentage-based error. "
        "Using multiple metrics provides a comprehensive evaluation."
    )

    if st.button("Run Model Evaluation", key="eval_btn"):
        st.session_state["show_eval"] = True

    if st.session_state.get("show_eval", False):
        preds = np.clip(load_model().predict(X_test_ts), 0, None)

        mae = mean_absolute_error(y_test_ts, preds)
        rmse = np.sqrt(mean_squared_error(y_test_ts, preds))
        r2 = r2_score(y_test_ts, preds)
        mape = compute_mape(y_test_ts, preds)

        st.subheader("RandomForest — Holdout Evaluation")

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("MAE", f"{mae:,.2f} W")
        col_m2.metric("RMSE", f"{rmse:,.2f} W")
        col_m3.metric("R² Score", f"{r2:.4f}")
        col_m4.metric("MAPE", f"{mape:.1f}%")

        fig_met, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        fig_met.patch.set_facecolor("#0D1117")
        labels = ["MAE", "RMSE", "R²"]
        values = [mae, rmse, r2]
        colors = [C_RF, C_ACCENT, C_ACTUAL]

        for i, ax in enumerate(axes):
            ax.bar([labels[i]], [values[i]], color=colors[i], alpha=0.9,
                   edgecolor="white", width=0.4)
            fmt = f'{values[i]:.4f}' if i == 2 else f'{values[i]:,.1f}'
            ax.text(0, values[i], fmt, ha='center', va='bottom', fontsize=10,
                    fontweight="bold", color="#E5E7EB")
            ax.set_title(labels[i], fontsize=11, fontweight="bold", color="#E5E7EB")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color("#374151")
            ax.spines["left"].set_color("#374151")
            ax.tick_params(colors="#9CA3AF")
            ax.set_facecolor("#111827")
        plt.tight_layout()
        st.pyplot(fig_met)

        if training_log and "cv_metrics_k5" in training_log:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.subheader("Cross-Validation (TimeSeriesSplit, k=5)")

            concept_note(
                "<strong>Concept — Cross-Validation:</strong> "
                "Instead of a single split, k-fold cross-validation trains and evaluates the model "
                "k times on different partitions. <strong>TimeSeriesSplit</strong> respects temporal "
                "order, preventing future data from leaking into training."
            )

            cv = training_log["cv_metrics_k5"]
            col_cv1, col_cv2, col_cv3, col_cv4 = st.columns(4)
            col_cv1.metric("CV MAE", f"{cv['MAE_mean']:,.1f} ± {cv['MAE_std']:,.1f}")
            col_cv2.metric("CV RMSE", f"{cv['RMSE_mean']:,.1f} ± {cv['RMSE_std']:,.1f}")
            col_cv3.metric("CV R²", f"{cv['R2_mean']:.4f} ± {cv['R2_std']:.4f}")
            col_cv4.metric("CV MAPE", f"{cv['MAPE_mean']:.1f} ± {cv['MAPE_std']:.1f}%")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("Actual vs Predicted")

        n_show = 300

        fig_ap, ax_ap = plt.subplots(figsize=(14, 5))
        fig_ap.patch.set_facecolor("#0D1117")
        ax_ap.plot(range(n_show), y_test_ts.values[:n_show], color=C_ACTUAL,
                   label="Actual", linewidth=1.5, alpha=0.8)
        ax_ap.plot(range(n_show), preds[:n_show], color=C_RF,
                   label="Predicted (RF)", linewidth=1.5, alpha=0.8)
        style_plot(ax_ap, "Actual vs Predicted Solar Power",
                   xlabel="Test Sample Index", ylabel="DC Power (W)")
        ax_ap.legend(loc="upper right", fontsize=9, facecolor="#111827",
                     edgecolor="#374151", labelcolor="#E5E7EB")
        plt.tight_layout()
        st.pyplot(fig_ap)

        concept_note(
            "<strong>Concept — Regression Evaluation:</strong> "
            "The closer the predicted line follows the actual line, the better the model. "
            "Large gaps indicate regions where the model struggles — typically low-light or "
            "transitional hours."
        )

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("Scatter Plot — Actual vs Predicted")

        fig_scat, ax_s = plt.subplots(figsize=(7, 5))
        fig_scat.patch.set_facecolor("#0D1117")
        ax_s.scatter(y_test_ts.values, preds, alpha=0.15, s=8, color=C_RF, edgecolors="none")
        max_val = max(y_test_ts.max(), preds.max())
        ax_s.plot([0, max_val], [0, max_val], color=C_ACTUAL, linestyle="--",
                  linewidth=1.5, label="Perfect Prediction")
        style_plot(ax_s, f"RandomForest (R²={r2:.4f})", xlabel="Actual (W)", ylabel="Predicted (W)")
        ax_s.legend(fontsize=9, facecolor="#111827", edgecolor="#374151", labelcolor="#E5E7EB")
        plt.tight_layout()
        st.pyplot(fig_scat)

        st.caption("Points on the diagonal = perfect predictions. "
                   "Tighter cluster = better model performance.")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("Residual Analysis")

        errors = y_test_ts.values - preds

        fig_res, (ax_r1, ax_r2) = plt.subplots(1, 2, figsize=(14, 4.5))
        fig_res.patch.set_facecolor("#0D1117")

        ax_r1.scatter(preds, errors, alpha=0.12, s=8, color=C_RF, edgecolors="none")
        ax_r1.axhline(0, color=C_ACTUAL, linestyle="--", linewidth=1.5, alpha=0.7)
        style_plot(ax_r1, "Residuals vs Predicted", xlabel="Predicted (W)", ylabel="Residual (W)")

        ax_r2.hist(errors, bins=60, color=C_RF, alpha=0.75, edgecolor="white", linewidth=0.5)
        ax_r2.axvline(0, color=C_ACTUAL, linestyle="--", linewidth=1.5, alpha=0.7)
        ax_r2.axvline(np.mean(errors), color="#E5E7EB", linestyle=":", linewidth=1, alpha=0.5,
                      label=f"Mean={np.mean(errors):,.0f}")
        style_plot(ax_r2, "Error Distribution", xlabel="Error (W)", ylabel="Frequency")
        ax_r2.legend(fontsize=9, facecolor="#111827", edgecolor="#374151", labelcolor="#E5E7EB")

        plt.tight_layout()
        st.pyplot(fig_res)

        concept_note(
            "<strong>Concept — Residual Analysis:</strong> "
            "A well-fitted model shows residuals scattered randomly around zero (low bias). "
            "A narrow error distribution means low variance. Patterns in residuals indicate "
            "systematic errors the model has not captured."
        )

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("Feature Importance — What Drives Predictions?")

        concept_note(
            "<strong>Concept — Decision Tree Feature Importance:</strong> "
            "Tree-based models rank features by how much they reduce <strong>Gini impurity</strong> "
            "across all splits. Features causing the largest reduction are most important. "
            "This provides <strong>model explainability</strong>."
        )

        imp = load_model().feature_importances_

        fig_imp, ax_imp = plt.subplots(figsize=(10, 4.5))
        fig_imp.patch.set_facecolor("#0D1117")
        bars = ax_imp.barh(FEATURES, imp, color=C_RF, alpha=0.9, edgecolor="white")
        style_plot(ax_imp, "Feature Importance — Gini-Based Ranking", xlabel="Importance Score")
        ax_imp.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig_imp)

        st.info("IRRADIATION is the dominant feature — as expected from solar physics. "
                "Power output is proportional to irradiance received by the panels.")


# ══════════════════════════════════════════════════════
# TAB 4 — FORECAST
# ══════════════════════════════════════════════════════
with tab4:
    st.header("Future Solar Forecast")

    concept_note(
        "<strong>Concept — Regression for Forecasting:</strong> "
        "By varying input features (hour, irradiation, temperature), we simulate future "
        "conditions and use the trained model to predict power output. This demonstrates "
        "the practical real-world application of supervised learning."
    )

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        fc_source = st.number_input("Inverter ID", 0, key="fc_source")
        fc_ambient = st.number_input("Ambient Temp (°C)", value=28.0, key="fc_amb")
        fc_module = st.number_input("Module Temp (°C)", value=40.0, key="fc_mod")
    with col_f2:
        fc_irradiation = st.number_input("Base Irradiation (0–1.2)", value=0.7, key="fc_irr")
        fc_hour = st.slider("Starting Hour", 0, 23, 8, key="fc_hour")
        fc_month = st.slider("Month", 1, 12, 5, key="fc_month")

    future_hours = st.slider("Forecast horizon (hours)", 1, 24, 12, key="fc_horizon")

    if st.button("Run Forecast", type="primary", key="fc_run"):

        preds_list = []
        hours_list = []

        season_factor = {
            1: 0.6, 2: 0.7, 3: 0.9, 4: 1.0, 5: 1.1, 6: 1.15,
            7: 1.1, 8: 1.0, 9: 0.9, 10: 0.8, 11: 0.7, 12: 0.6
        }

        for h in range(future_hours):
            future_hour = (fc_hour + h) % 24
            hours_list.append(future_hour)

            if 6 <= future_hour <= 18:
                solar_factor = np.sin(np.pi * (future_hour - 6) / 12)
            else:
                solar_factor = 0

            solar_factor *= season_factor.get(fc_month, 1.0)

            future_irradiation = max(fc_irradiation * solar_factor, 0)
            temp_variation = 0.9 + 0.25 * solar_factor
            future_module_temp = fc_module * temp_variation
            future_ambient_temp = fc_ambient * (0.95 + 0.1 * solar_factor)

            future_input = pd.DataFrame([[
                fc_source, future_ambient_temp, future_module_temp,
                future_irradiation, future_hour, fc_month,
            ]], columns=FEATURES)

            p = max(0, load_model().predict(future_input)[0])

            if future_hour < 6 or future_hour > 18:
                p = 0

            preds_list.append(p)

        st.subheader("Forecast Curve")

        fig_fc, ax_fc = plt.subplots(figsize=(14, 5))
        fig_fc.patch.set_facecolor("#0D1117")
        ax_fc.plot(hours_list, preds_list, color=C_RF,
                   linewidth=2.5, marker="o", markersize=6, label="Forecast", zorder=3)
        ax_fc.fill_between(range(len(hours_list)), preds_list,
                           alpha=0.15, color=C_RF)
        style_plot(ax_fc, "Solar Power Forecast — RandomForest",
                   xlabel="Hour of Day", ylabel="DC Power (W)")
        ax_fc.set_xticks(range(len(hours_list)))
        ax_fc.set_xticklabels(hours_list)
        ax_fc.legend(fontsize=10, facecolor="#111827", edgecolor="#374151", labelcolor="#E5E7EB")
        plt.tight_layout()
        st.pyplot(fig_fc)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        col_i1, col_i2 = st.columns(2)
        col_i1.metric("Peak Forecast", f"{max(preds_list):,.0f} W")
        col_i2.metric("Peak Hour", f"{hours_list[np.argmax(preds_list)]}:00")

        best_hours = sorted(
            list(zip(hours_list, preds_list)),
            key=lambda x: x[1], reverse=True
        )[:3]
        st.success(f"Peak generation hours: {[h[0] for h in best_hours]}")

        forecast_df = pd.DataFrame({
            "Hour": hours_list,
            "Forecast_W": [round(p, 2) for p in preds_list],
        })
        csv_fc = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Forecast Results", csv_fc,
                           "solar_forecast_results.csv", "text/csv")


# ══════════════════════════════════════════════════════
# TAB 5 — LOGS & EXPORT
# ══════════════════════════════════════════════════════
with tab5:
    st.header("Training Logs & Data Export")

    if training_log:
        st.subheader("Training Log")

        concept_note(
            "<strong>Concept — Reproducibility:</strong> "
            "Logging hyperparameters, feature lists, and metrics ensures that "
            "the training process is <strong>reproducible</strong> — a key requirement "
            "in both academic research and practical ML systems."
        )

        col_l1, col_l2 = st.columns(2)
        with col_l1:
            st.markdown("**Model Configuration**")
            st.json(training_log.get("hyperparameters", {}))
            st.markdown("**Features Used**")
            st.write(training_log.get("features", []))

        with col_l2:
            st.markdown("**Holdout Metrics**")
            st.json(training_log.get("holdout_metrics", {}))
            st.markdown("**Cross-Validation Metrics (k=5)**")
            st.json(training_log.get("cv_metrics_k5", {}))

        st.markdown("**Dataset Info**")
        st.json(training_log.get("dataset", {}))

        log_json = json.dumps(training_log, indent=2).encode("utf-8")
        st.download_button("Download Training Log (JSON)", log_json,
                           "training_log.json", "application/json")
    else:
        st.warning("No training log found. Run `python -m src.modeling.train` to generate it.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("Export Test Set Results")

    if st.button("Generate Test Set Export", key="export_btn"):
        st.session_state["show_export"] = True

    if st.session_state.get("show_export", False):
        preds_export = np.clip(load_model().predict(X_test_ts), 0, None)

        export_df = X_test_ts.copy()
        export_df["Actual_DC_POWER"] = y_test_ts.values
        export_df["Predicted"] = preds_export
        export_df["Error"] = y_test_ts.values - preds_export
        export_df["Abs_Error"] = np.abs(y_test_ts.values - preds_export)

        st.write(f"Test set size: **{len(export_df):,}** samples. Features are now included so this file can be reused in the Batch Prediction tool!")
        st.dataframe(export_df.head(20), use_container_width=True)

        csv_export = export_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Test Results (CSV)", csv_export,
                           "test_features_and_predictions.csv", "text/csv")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("Summary Statistics")

        summary = pd.DataFrame({
            "Statistic": ["Mean", "Median", "Std Dev", "Min", "Max", "25th %ile", "75th %ile"],
            "Actual": [
                f"{y_test_ts.mean():,.1f}", f"{y_test_ts.median():,.1f}",
                f"{y_test_ts.std():,.1f}", f"{y_test_ts.min():,.1f}",
                f"{y_test_ts.max():,.1f}", f"{y_test_ts.quantile(0.25):,.1f}",
                f"{y_test_ts.quantile(0.75):,.1f}",
            ],
            "Predicted": [
                f"{preds_export.mean():,.1f}", f"{np.median(preds_export):,.1f}",
                f"{preds_export.std():,.1f}", f"{preds_export.min():,.1f}",
                f"{preds_export.max():,.1f}", f"{np.percentile(preds_export, 25):,.1f}",
                f"{np.percentile(preds_export, 75):,.1f}",
            ],
        })

        st.dataframe(summary.set_index("Statistic"), use_container_width=True)


# ══════════════════════════════════════════════════════
# TAB 6 — AI GRID OPTIMIZER (RAG)
# ══════════════════════════════════════════════════════
with tab6:
    st.header("Agentic AI Grid Optimization Assistant")
    concept_note(
        "<strong>Concept — Agentic RAG:</strong> "
        "This assistant runs as a separate layer on top of your ML forecaster. "
        "It builds a solar forecast, analyzes variability risk, retrieves grid guidelines, "
        "and produces a structured optimization report."
    )

    with st.expander("Scenario Inputs & Operator Goal", expanded=True):
        with st.form("rag_scenario_form"):
            col_rg1, col_rg2, col_rg3 = st.columns(3)
            with col_rg1:
                st.markdown("##### Environment & Solar")
                rag_source_key = st.number_input("Inverter ID", min_value=0, value=0, key="rag_src")
                rag_ambient = st.number_input("Ambient Temp (°C)", value=28.0, key="rag_amb")
                rag_module = st.number_input("Module Temp (°C)", value=40.0, key="rag_mod")
                rag_irr = st.number_input("Base Irradiation", min_value=0.0, max_value=1.5, value=0.75, key="rag_irr")
                rag_month = st.slider("Month", 1, 12, 6, key="rag_month")
            with col_rg2:
                st.markdown("##### Grid & Load Profile")
                rag_start = st.slider("Starting Hour", 0, 23, 8, key="rag_start")
                rag_horizon = st.slider("Forecast Horizon", 6, 24, 12, key="rag_horiz")
                rag_demand = st.number_input("Average Grid Demand (kW)", min_value=1.0, value=22.0, key="rag_dem")
                rag_reserve = st.slider("Reserve Margin (%)", 5, 40, 15, key="rag_res")
                rag_crit = st.slider("Critical Load Share", 0.2, 1.0, 0.6, 0.05, key="rag_crit")
            with col_rg3:
                st.markdown("##### Storage Strategy")
                rag_bat_cap = st.number_input("Battery Capacity (kWh)", min_value=0.0, value=50.0, key="rag_bat_cap")
                rag_bat_soc = st.slider("Battery SOC (%)", 0, 100, 55, key="rag_soc")
                rag_bat_pow = st.number_input("Battery Power Limit (kW)", min_value=0.0, value=15.0, key="rag_bat_pow")
            
            question = st.text_area(
                "Operator Goal",
                value=(
                    "Generate a grid optimization plan that minimizes deficit risk, uses storage well, "
                    "and shifts flexible loads into strong solar periods."
                ),
                height=70,
            )

            # The button is now inside the form
            submitted = st.form_submit_button("Run Agentic RAG Optimization", type="primary")

    scenario = {
        "source_key": rag_source_key,
        "ambient_temperature": rag_ambient,
        "module_temperature": rag_module,
        "base_irradiation": rag_irr,
        "start_hour": rag_start,
        "month": rag_month,
        "forecast_horizon_hours": rag_horizon,
        "grid_demand_kw": rag_demand,
        "reserve_margin_pct": rag_reserve,
        "critical_load_share": rag_crit,
        "battery_capacity_kwh": rag_bat_cap,
        "battery_soc_pct": rag_bat_soc,
        "battery_power_kw": rag_bat_pow,
    }

    if submitted:
        with st.spinner("Analyzing forecast, risks, and retrieved grid guidance..."):
            st.session_state.rag_state = run_agentic_grid_optimizer(question=question, scenario=scenario)

    if "rag_state" in st.session_state:
        state = st.session_state.rag_state
        report = state["optimization_report"]
        forecast_df = pd.DataFrame(state["forecast_rows"])
        references_df = pd.DataFrame(state["supporting_references"])

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        top1, top2, top3, top4 = st.columns(4)
        top1.metric("Risk Level", state["risk_analysis"]["risk_level"])
        top2.metric("Risk Score", f"{state['risk_analysis']['risk_score']}/100")
        top3.metric("Peak Solar", f"{report['forecast_summary']['peak_generation_kw']:.2f} kW")
        top4.metric("Coverage Ratio", f"{report['forecast_summary']['coverage_ratio']:.0%}")

        rtab1, rtab2, rtab3, rtab4 = st.tabs(
            ["Optimization Report", "Forecast & Risk", "Retrieved Guidelines", "Agent Trace"]
        )

        with rtab1:
            st.markdown(state["report_markdown"])
            
            st.subheader("AI Reasoning")
            if "llm_reasoning" in state:
                st.info(state["llm_reasoning"])
                
            st.subheader("Ask AI")
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            
            # Display chat history natively
            for q, a in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(q)
                with st.chat_message("assistant"):
                    st.write(a)

            # Accept user input
            if user_q := st.chat_input("Ask about this scenario..."):
                with st.chat_message("user"):
                    st.write(user_q)
                
                with st.chat_message("assistant"):
                    import os
                    from groq import Groq
                    try:
                        with st.spinner("Thinking..."):
                            
                            groq_key = os.getenv("GROQ_API_KEY")
                            try:
                                if not groq_key and hasattr(st, "secrets"):
                                    groq_key = st.secrets.get("GROQ_API_KEY")
                            except:
                                pass
                            client = Groq(api_key=groq_key)

                            prompt = f"Context:\n{state['optimization_report']}\n\nUser question: {user_q}\nRespond clearly and concisely with formatting."
                            response = client.chat.completions.create(
                                model="llama-3.1-8b-instant",
                                messages=[{"role": "user", "content": prompt}],
                            )
                            answer = response.choices[0].message.content
                            st.write(answer)
                            st.session_state.chat_history.append((user_q, answer))
                    except Exception as e:
                        st.error(f"Chatbot failed! Make sure GROQ_API_KEY is set. Error: {str(e)}")

            st.download_button(
                "Download Report (Markdown)",
                data=state["report_markdown"].encode("utf-8"),
                file_name="agentic_grid_optimization_report.md",
                mime="text/markdown",
            )
            st.download_button(
                "Download Report (JSON)",
                data=json.dumps(report, indent=2).encode("utf-8"),
                file_name="agentic_grid_optimization_report.json",
                mime="application/json",
            )

        with rtab2:
                st.subheader("Hourly Forecast Table")
                
                # Only show columns that actually exist
                desired_cols = [
                    "hour", "predicted_kw", "demand_kw", "net_margin_kw",
                    "ramp_delta_kw", "historical_mean_w", "historical_variability_ratio",
                ]
                available_cols = [c for c in desired_cols if c in forecast_df.columns]
                
                st.dataframe(
                    forecast_df[available_cols],
                    use_container_width=True,
                )

        with rtab3:
            st.caption(f"Retriever backend: {state.get('retriever_backend', 'tfidf-cosine')}")
            st.dataframe(references_df, use_container_width=True)
            for item in state["retrieved_guidelines"]:
                with st.expander(f"{item['title']} ({item['id']})"):
                    st.write(item["content"])
                    st.caption(item["source"])

        with rtab4:
            st.dataframe(pd.DataFrame(state["trace"]), use_container_width=True)
            st.json(
                {
                    "scenario": state["scenario"],
                    "forecast_summary": state["forecast_summary"],
                    "risk_analysis": state["risk_analysis"],
                }
            )