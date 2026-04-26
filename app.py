import streamlit as st
from inference import MIN_SYMPTOMS, load_resources, predict_from_symptoms

# ----------------- Helper Functions -----------------
@st.cache_resource(show_spinner=False)
def get_cached_resources():
    return load_resources()

def format_option(option):
    if option == "None":
        return "Select a symptom..."
    return option.replace("_", " ").title()

# ----------------- Custom CSS -----------------
def local_css():
    st.markdown("""
        <style>
        html, body, [class*="css"], .stApp {
            font-family: Inter, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        }

        .stApp {
            background: linear-gradient(180deg, #f8fbff 0%, #eef5ff 100%);
            background-attachment: fixed;
        }

        h1, h2, h3, h4 {
            color: #0a2540 !important;
            font-weight: 700;
        }

        .stButton > button {
            background: linear-gradient(135deg, #2451ff, #3f8cff);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.7rem 1rem;
            font-weight: 600;
            width: 100%;
        }

        .stButton > button:hover {
            filter: brightness(1.05);
        }

        .result-card {
            background: #ffffff;
            border: 1px solid #dce6f6;
            border-radius: 12px;
            padding: 1rem 1.2rem;
            box-shadow: 0 4px 12px rgba(36, 81, 255, 0.08);
            margin-bottom: 1rem;
        }

        .top-condition-row {
            background: #ffffff;
            border: 1px solid #dce6f6;
            border-radius: 10px;
            padding: 0.55rem 0.75rem;
            margin-top: 0.45rem;
            margin-bottom: 0.35rem;
        }

        .top-condition-title {
            color: #0a2540;
            font-size: 0.95rem;
            font-weight: 700;
            line-height: 1.2;
        }

        .top-condition-sub {
            color: #4f5d75;
            font-size: 0.82rem;
            font-weight: 600;
            margin-top: 0.15rem;
        }

        .badge {
            display: inline-block;
            border-radius: 999px;
            padding: 0.2rem 0.7rem;
            font-size: 0.75rem;
            font-weight: 600;
            margin-left: 0.35rem;
        }

        .badge-low { background: #fee2e2; color: #991b1b; }
        .badge-moderate { background: #fef3c7; color: #92400e; }
        .badge-high { background: #dcfce7; color: #166534; }

        .subtitle {
            color: #4f5d75;
            margin-top: -0.35rem;
            margin-bottom: 1rem;
        }

        .footer {
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #dce6f6;
            color: #6b7280;
            font-size: 0.82rem;
            text-align: center;
        }

        .stWarning, div[data-baseweb="notification"] {
            border-radius: 10px !important;
        }

        .stWarning {
            background-color: #fff7e6 !important;
            border: 1px solid #f3c46b !important;
            color: #7c4a03 !important;
        }

        .stWarning p, .stWarning div {
            color: #7c4a03 !important;
            font-weight: 600;
        }

        .stInfo {
            background-color: #e8f2ff !important;
            border: 1px solid #96b8ef !important;
            color: #123a75 !important;
            border-radius: 10px !important;
        }

        .stInfo p, .stInfo div {
            color: #123a75 !important;
        }

        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)

# ----------------- App Layout -----------------
st.set_page_config(page_title="Health Symptom Analyzer", page_icon="🏥", layout="wide")

# Inject Custom CSS
local_css()

# --- Header ---
st.title("Health Symptom Analyzer")
st.markdown("<p class='subtitle'>Symptom-based triage assistance powered by a calibrated ensemble model.</p>", unsafe_allow_html=True)

try:
    resources = get_cached_resources()
    model = resources["model"]
    disease_names = resources["disease_names"]
    model_feature_list = resources["feature_names"]
    symptom_synonyms = resources["symptom_synonyms"]
    confidence_threshold = resources["confidence_threshold"]
    margin_threshold = resources["top2_margin_threshold"]
    model_version = resources["model_version"]
    all_symptoms = ["None"] + model_feature_list
except Exception as exc:  # pragma: no cover - Streamlit runtime guard
    model = None
    disease_names = []
    model_feature_list = []
    symptom_synonyms = {}
    confidence_threshold = 0.45
    margin_threshold = 0.08
    model_version = "unknown"
    all_symptoms = ["None"]
    st.error(f"Failed to load resources: {exc}")

with st.sidebar:
    st.markdown("### Input Controls")
    min_required = st.slider("Minimum symptoms required", min_value=2, max_value=6, value=MIN_SYMPTOMS, step=1)
    top_k = st.slider("Top predictions to display", min_value=1, max_value=5, value=3, step=1)
    st.markdown("### Notes")
    st.caption("Use precise symptom terms for best results. The model returns ranked suggestions, not a medical diagnosis.")
    st.caption(f"Model version: {model_version}")

left_col, right_col = st.columns([1.2, 1], gap="large")

with left_col:
    st.markdown("### Symptom Selection")
    symptom_inputs = []
    for i in range(1, 6):
        val = st.selectbox(
            f"Symptom {i}",
            options=all_symptoms,
            index=0,
            key=f"s{i}",
            format_func=format_option,
        )
        symptom_inputs.append(val)
    run_prediction = st.button("Run Analysis", type="primary", use_container_width=True)

# --- Prediction Logic ---
with right_col:
    st.markdown("### Prediction Result")
    if not run_prediction:
        st.info("Select symptoms on the left and run analysis.")

if run_prediction:
    if model is None:
        st.error("❌ Model is not available. Run `python main.py` to train/load models first.")
        st.stop()
    
    selected = [symptom for symptom in symptom_inputs if symptom != "None"]
    result = predict_from_symptoms(
        symptoms=selected,
        model=model,
        feature_names=model_feature_list,
        disease_names=disease_names,
        symptom_synonyms=symptom_synonyms,
        min_symptoms=min_required,
        inconclusive_threshold=confidence_threshold,
        top2_margin_threshold=margin_threshold,
        model_version=model_version,
    )

    with right_col:
        if not result["ok"]:
            st.error(
                f"{result['message']} (received {len(result['selected_symptoms'])} valid symptoms)"
            )
            if result["ignored_symptoms"]:
                st.caption(f"Ignored symptoms: {', '.join(result['ignored_symptoms'])}")
        else:
            band = result["confidence_band"]
            badge_class = f"badge-{band}"
            confidence_pct = result["confidence"] * 100
            relative_pct = result["relative_confidence_top3"] * 100

            status_text = "Inconclusive" if result["is_inconclusive"] else result["predicted_disease"]
            st.markdown(
                f"<div class='result-card'><h4>{status_text}<span class='badge {badge_class}'>{band.upper()}</span></h4>"
                f"<p>Absolute confidence: <b>{confidence_pct:.1f}%</b><br/>"
                f"Relative confidence (within top-{top_k}): <b>{relative_pct:.1f}%</b></p></div>",
                unsafe_allow_html=True,
            )

            metric_col_1, metric_col_2 = st.columns(2)
            metric_col_1.metric("Absolute confidence", f"{confidence_pct:.1f}%")
            metric_col_2.metric("Top-2 margin", f"{result['top2_margin']*100:.1f}%")

            st.caption(
                f"Thresholds: confidence >= {confidence_threshold:.2f}, margin >= {margin_threshold:.2f}"
            )

            st.markdown(f"#### Top {top_k} Conditions")
            for prediction in result["top_predictions"][:top_k]:
                st.markdown(
                    "<div class='top-condition-row'>"
                    f"<div class='top-condition-title'>{prediction['disease']}</div>"
                    f"<div class='top-condition-sub'>{prediction['confidence']*100:.1f}% confidence</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
                st.progress(float(prediction["confidence"]))
                st.write("")

            if result["ignored_symptoms"]:
                st.warning(f"Ignored symptoms: {', '.join(result['ignored_symptoms'])}")

            if result["requires_clinician_review"]:
                st.warning(
                    f"Escalation flag: clinician review recommended "
                    f"({result['escalation_reason'].replace('_', ' ')})."
                )

            st.info(
                "This system provides triage-oriented ML predictions and should be used with professional medical judgment."
            )

# --- Footer ---
st.markdown(
    """
    <div class='footer'>
        <p>Applied Machine Learning Project • Fall 2025</p>
        <p><strong>Developed by:</strong> Dax, Harsh, Charanish</p>
    </div>
    """, 
    unsafe_allow_html=True
)