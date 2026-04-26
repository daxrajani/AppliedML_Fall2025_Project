import os
from typing import List

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Import tuned model functions
from models.knn_model import get_knn_model
from models.naive_bayes_model import get_nb_model
from models.dt_model import get_dt_model
from models.rf_model import get_rf_model
from models.svm_model import get_svm_model
from models.logistic_model import get_logistic_model
from models.xgb_model import get_xgb_model 

MODEL_DIR_MAIN = "saved_models_main"
os.makedirs(MODEL_DIR_MAIN, exist_ok=True)

model_paths = {
    'KNN': os.path.join(MODEL_DIR_MAIN, 'knn.pkl'),
    'Naive Bayes': os.path.join(MODEL_DIR_MAIN, 'nb.pkl'),
    'Decision Tree': os.path.join(MODEL_DIR_MAIN, 'dt.pkl'),
    'Random Forest': os.path.join(MODEL_DIR_MAIN, 'rf.pkl'),
    'SVM': os.path.join(MODEL_DIR_MAIN, 'svm.pkl'),
    'Logistic Regression': os.path.join(MODEL_DIR_MAIN, 'lr.pkl'),
    'XGBoost': os.path.join(MODEL_DIR_MAIN, 'xgb.pkl')
}
voting_path = os.path.join(MODEL_DIR_MAIN, "voting_clf.pkl")

symptoms_list = [
    'itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills','joint_pain',
    'stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting','burning_micturition','spotting_ urination','fatigue',
    'weight_gain','anxiety','cold_hands_and_feets','mood_swings','weight_loss','restlessness','lethargy','patches_in_throat',
    'irregular_sugar_level','cough','high_fever','sunken_eyes','breathlessness','sweating','dehydration','indigestion',
    'headache','yellowish_skin','dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain','constipation',
    'abdominal_pain','diarrhoea','mild_fever','yellow_urine','yellowing_of_eyes','acute_liver_failure','fluid_overload',
    'swelling_of_stomach','swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
    'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs','fast_heart_rate',
    'pain_during_bowel_movements','pain_in_anal_region','bloody_stool','irritation_in_anus','neck_pain','dizziness','cramps',
    'bruising','obesity','swollen_legs','swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
    'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips','slurred_speech','knee_pain','hip_joint_pain',
    'muscle_weakness','stiff_neck','swelling_joints','movement_stiffness','spinning_movements','loss_of_balance','unsteadiness','weakness_of_one_body_side',
    'loss_of_smell','bladder_discomfort','foul_smell_of urine','continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain','abnormal_menstruation','dischromic _patches',
    'watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum','rusty_sputum','lack_of_concentration','visual_disturbances',
    'receiving_blood_transfusion','receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption',
    'fluid_overload','blood_in_sputum','prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
    'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze'
]

def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    return [item for item in items if not (item in seen or seen.add(item))]


def _prepare_dataframe() -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    try:
        df = pd.read_csv("Prototype.csv")
    except FileNotFoundError:
        raise FileNotFoundError("Error: 'Prototype.csv' not found. Please place it in the project directory.")

    df.columns = [column.strip() for column in df.columns]
    if "prognosis" not in df.columns:
        raise ValueError("Expected a 'prognosis' column in Prototype.csv")

    df["prognosis"] = df["prognosis"].astype(str).str.strip()
    df.dropna(subset=["prognosis"], inplace=True)
    df = df[df["prognosis"] != ""]

    expected_symptoms = _dedupe(symptoms_list)
    available_symptoms = [symptom for symptom in expected_symptoms if symptom in df.columns]
    missing_symptoms = [symptom for symptom in expected_symptoms if symptom not in df.columns]
    if missing_symptoms:
        print(f"Warning: {len(missing_symptoms)} configured symptoms missing from dataset.")

    # Ensure binary numeric values.
    feature_df = df[available_symptoms].fillna(0)
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    feature_df = (feature_df > 0).astype(int)

    # Remove features with no variance to stabilize model probabilities.
    non_constant_features = [column for column in feature_df.columns if feature_df[column].nunique() > 1]
    dropped = len(feature_df.columns) - len(non_constant_features)
    if dropped:
        print(f"Dropped {dropped} constant symptom features.")
    feature_df = feature_df[non_constant_features]

    le = LabelEncoder()
    labels = pd.Series(le.fit_transform(df["prognosis"]), name="prognosis")
    disease_names = list(le.classes_)

    with open("available_symptoms.txt", "w", encoding="utf-8") as symptom_file:
        symptom_file.write("\n".join(non_constant_features))
    with open("disease_names.txt", "w", encoding="utf-8") as disease_file:
        disease_file.write("\n".join(disease_names))

    print(f"Prepared dataset with {len(non_constant_features)} symptoms and {len(disease_names)} diseases.")
    return feature_df, labels, non_constant_features, disease_names


def train_pipeline():
    X, y, active_symptoms, diseases_list = _prepare_dataframe()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Data split: {len(X_train)} train / {len(X_test)} test samples.")

    models = {}
    model_getters = {
        "KNN": get_knn_model,
        "Naive Bayes": get_nb_model,
        "Decision Tree": get_dt_model,
        "Random Forest": get_rf_model,
        "SVM": get_svm_model,
        "Logistic Regression": get_logistic_model,
        "XGBoost": get_xgb_model,
    }

    print("\n--- Training base models ---")
    for name, getter_func in model_getters.items():
        path = model_paths[name]
        print(f"Training {name}...")
        model = getter_func(X_train, y_train)
        models[name] = model
        joblib.dump(model, path)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred) * 100
        print(f"{name:20s} accuracy: {acc:.2f}%")

    print("\n--- Training calibrated ensemble ---")
    estimators = [
        ("KNN", models["KNN"]),
        ("Naive Bayes", models["Naive Bayes"]),
        ("Decision Tree", models["Decision Tree"]),
        ("Random Forest", models["Random Forest"]),
        ("SVM", models["SVM"]),
        ("Logistic Regression", models["Logistic Regression"]),
        ("XGBoost", models["XGBoost"]),
    ]
    weights = [1, 1, 1, 4, 2, 2, 3]

    ensemble = VotingClassifier(estimators=estimators, voting="soft", weights=weights)
    calibrated_ensemble = CalibratedClassifierCV(ensemble, method="sigmoid", cv=3)
    calibrated_ensemble.fit(X_train, y_train)
    joblib.dump(calibrated_ensemble, voting_path)

    y_pred = calibrated_ensemble.predict(X_test)
    y_proba = calibrated_ensemble.predict_proba(X_test)
    ensemble_acc = accuracy_score(y_test, y_pred) * 100
    ensemble_loss = log_loss(y_test, y_proba)
    mean_top_conf = float(y_proba.max(axis=1).mean()) * 100
    print(f"Calibrated ensemble accuracy: {ensemble_acc:.2f}%")
    print(f"Calibrated ensemble log-loss: {ensemble_loss:.4f}")
    print(f"Average top-class confidence: {mean_top_conf:.2f}%")

    return calibrated_ensemble, models, active_symptoms, diseases_list


# ----------------- Prediction Application -----------------
def predict_disease(user_input_list):
    """
    Takes user input, cleans it, and predicts the disease using the Voting Classifier.
    Also provides confidence scores and fallback logic.
    """
    # 1. Input Cleaning
    valid_symptoms = []
    ignored_symptoms = []
    
    for s in user_input_list:
        # Standardize input to match dataset format (lowercase, underscores)
        clean_s = s.strip().lower().replace(' ', '_').replace("'", "")
        if clean_s in symptoms_list:
            valid_symptoms.append(clean_s)
        elif clean_s != "":
            ignored_symptoms.append(clean_s)
            
    print(f"\n✅ Accepted Symptoms: {valid_symptoms}")
    if ignored_symptoms:
        print(f"⚠️ Ignored (Typo?): {ignored_symptoms}")

    # 2. Safety Check
    if len(valid_symptoms) < 3:
        print("\n⚠️  WARNING: Low information. Please provide at least 3 valid symptoms.")
        
    # 3. Prepare Input Vector
    input_vector = [1 if symptom in valid_symptoms else 0 for symptom in symptoms_list]
    input_df = pd.DataFrame([input_vector], columns=symptoms_list)

    # 4. Individual Model Opinions
    print("\n--- Individual Model Predictions ---")
    for name, model in models.items():
        pred_label = model.predict(input_df)[0]
        print(f"{name}: {diseases_list[pred_label]}")

    # 5. Ensemble Prediction with Confidence
    probas = voting_clf.predict_proba(input_df)[0]
    
    # Find the top 3 matches
    top3_indices = probas.argsort()[-3:][::-1]
    
    print(f"\n--- Ensemble Confidence (Soft Voting) ---")
    for i in top3_indices:
        disease = diseases_list[i]
        score = probas[i]
        if score > 0.01: # Only show meaningful probabilities
            print(f"  {score*100:.1f}%: {disease}")

    # 6. Final Decision
    max_confidence = probas[top3_indices[0]]
    final_disease = diseases_list[top3_indices[0]]

    # Threshold: If confidence is too low, don't guess blindly
    if max_confidence < 0.3:
        print(f"\n>>> RESULT: Inconclusive.")
        print(f"    The system is unsure ({max_confidence*100:.1f}%). Closest match: {final_disease}")
    else:
        print(f"\n>>> FINAL DIAGNOSIS: {final_disease} <<<")


# ----------------- Interactive Loop -----------------
if __name__ == "__main__":
    voting_clf, models, symptoms_list, diseases_list = train_pipeline()
    print("\n===============================================")
    print("      Health Symptom Analyzer")
    print("===============================================")
    print(f"Database: {len(diseases_list)} Diseases, {len(symptoms_list)} Symptoms.")
    print("Tip: Check 'available_symptoms.txt' for correct spelling.")
    
    while True:
        print("\n" + "-"*30)
        user_input = input("Enter symptoms (comma-separated) or 'q' to quit:\n> ")
        
        if user_input.lower() in ["exit", "quit", "q"]:
            print("\nStay healthy! Goodbye. 👋")
            break
        
        if user_input.strip():
            predict_disease(user_input.split(","))