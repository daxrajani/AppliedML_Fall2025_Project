import streamlit as st
import time

from inference import MIN_SYMPTOMS, load_resources, predict_from_symptoms

# ----------------- Helper Functions -----------------
@st.cache_resource(show_spinner=False)
def get_cached_resources():
    return load_resources()

def format_option(option):
    if option == "None":
        return "Select a symptom..."
    return option.replace("_", " ").title()

# ----------------- Custom CSS (Medical Light Theme) -----------------
def local_css():
    st.markdown("""
        <style>
        /* 1. Global Font & Colors */
        html, body, [class*="css"] {
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            color: #2c3e50;
        }

        /* 2. Main Background - Clean Medical White/Teal Gradient */
        .stApp {
            background: linear-gradient(to bottom right, #ffffff 0%, #e0f7fa 100%);
            background-attachment: fixed;
        }

        /* 3. Headers - Deep Medical Blue */
        h1, h2, h3 {
            color: #0277bd !important;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        p, label, .stMarkdown {
            color: #455a64 !important; /* Soft Slate Grey */
            font-weight: 500;
        }

        /* 4. Glassmorphism Inputs (The Box Itself) */
        .stSelectbox > div > div {
            background-color: rgba(255, 255, 255, 0.85) !important;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid #b3e5fc; /* Light Blue Border */
            border-radius: 12px;
            color: #01579b !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        }

        /* Hover effect on the box */
        .stSelectbox > div > div:hover {
            border: 1px solid #0288d1;
            box-shadow: 0 6px 12px rgba(2, 136, 209, 0.1);
        }

        /* 5. Dropdown Menu Items (The list that pops up) */
        ul[data-baseweb="menu"] {
            background-color: #ffffff !important;
            border: 1px solid #e1f5fe !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
            border-radius: 10px !important;
        }

        li[data-baseweb="option"] {
            color: #37474f !important;
        }

        li[data-baseweb="option"]:hover, li[aria-selected="true"] {
            background-color: #e1f5fe !important;
            color: #0277bd !important;
            font-weight: bold;
        }

        /* The selected value text in the closed box */
        .stSelectbox div[data-baseweb="select"] span {
            color: #01579b !important; 
            font-weight: 600;
        }
        
        /* 6. VISIBILITY FIX: Text color WHEN TYPING (Search) */
        .stSelectbox input {
            color: #01579b !important; 
            caret-color: #01579b !important;
        }
        
        /* 7. ARROW FIX: Force the Dropdown Arrow to be Blue */
        .stSelectbox div[data-baseweb="select"] svg {
            fill: #01579b !important;
            stroke: #01579b !important;
        }

        /* 8. Analyze Button */
        .stButton > button {
            background: linear-gradient(45deg, #0288d1, #26c6da);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 12px 24px;
            font-weight: bold;
            letter-spacing: 1px;
            box-shadow: 0 4px 10px rgba(2, 136, 209, 0.3);
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stButton > button:hover {
            background: linear-gradient(45deg, #0277bd, #00acc1);
            box-shadow: 0 6px 15px rgba(2, 136, 209, 0.4);
            transform: translateY(-2px);
        }

        /* 9. Success/Result Box */
        div[data-baseweb="notification"] {
            background-color: rgba(224, 247, 250, 0.95) !important;
            border: 1px solid #4dd0e1;
            color: #006064;
            border-radius: 12px;
        }
        
        /* 10. Disclaimer Box */
        .stAlert {
            background-color: rgba(227, 242, 253, 0.95) !important;
            border: 1px solid #64b5f6;
            color: #0d47a1;
            border-radius: 12px;
        }
        
        /* 11. Progress Bar */
        .stProgress > div > div > div > div {
            background-image: linear-gradient(45deg, #0288d1, #26c6da);
        }

        /* 12. Footer */
        .footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #cfd8dc;
            color: #90a4ae;
            font-size: 0.85rem;
            text-align: center;
        }

        /* 13. HIDE STREAMLIT DEFAULT HEADER & FOOTER */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Adjust top padding since header is gone */
        .block-container {
            padding-top: 2rem !important;
        }
        </style>
        """, unsafe_allow_html=True)

# ----------------- App Layout -----------------
st.set_page_config(page_title="Disease Prediction System", page_icon="🏥", layout="centered")

# Inject Custom CSS
local_css()

# --- Header ---
st.title("🏥 Health Symptom Analyzer")
st.markdown("### Describe Your Symptoms")
st.write("Please select up to 5 symptoms from the dropdowns below to receive a preliminary assessment.")

try:
    resources = get_cached_resources()
    model = resources["model"]
    disease_names = resources["disease_names"]
    model_feature_list = resources["feature_names"]
    symptom_synonyms = resources["symptom_synonyms"]
    all_symptoms = ["None"] + model_feature_list
except Exception as exc:  # pragma: no cover - Streamlit runtime guard
    model = None
    disease_names = []
    model_feature_list = []
    symptom_synonyms = {}
    all_symptoms = ["None"]
    st.error(f"Failed to load resources: {exc}")

# --- Input Section ---
symptom_inputs = []

with st.container():
    for i in range(1, 6):
        val = st.selectbox(
            f"Symptom {i}", 
            options=all_symptoms, 
            index=0, 
            key=f"s{i}", 
            format_func=format_option
        )
        symptom_inputs.append(val)

st.write("") # Spacer

# --- Prediction Logic ---
if st.button("Analyze Health Condition", type="primary", use_container_width=True):
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
    )

    if not result["ok"]:
        st.warning(
            f"⚠️ **Insufficient Data:** {result['message']} "
            f"(Selected {len(result['selected_symptoms'])} valid symptoms, minimum is {MIN_SYMPTOMS})."
        )
        if result["ignored_symptoms"]:
            st.caption(f"Ignored symptoms: {', '.join(result['ignored_symptoms'])}")
    else:
        # Prediction animation
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1)
        my_bar.empty()

        st.divider()
        if result["is_inconclusive"]:
            st.warning(
                f"### Assessment Result: **Inconclusive** "
                f"(confidence {result['confidence']*100:.1f}%)"
            )
        else:
            st.success(
                f"### Assessment Result: **{result['predicted_disease']}** "
                f"({result['confidence']*100:.1f}% confidence)"
            )

        st.markdown("#### Top 3 Possible Conditions")
        for prediction in result["top_predictions"]:
            st.write(f"- {prediction['disease']}: {prediction['confidence']*100:.1f}%")

        if result["ignored_symptoms"]:
            st.caption(f"Ignored symptoms: {', '.join(result['ignored_symptoms'])}")

        st.info(
            "⚠️ **Medical Disclaimer:** This ML tool is for educational triage support only and "
            "is not a diagnosis. Always consult a qualified medical professional."
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