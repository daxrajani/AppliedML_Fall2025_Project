import pandas as pd
import joblib  
import os      
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Import tuned model functions
from models.knn_model import get_knn_model
from models.naive_bayes_model import get_nb_model
from models.dt_model import get_dt_model
from models.rf_model import get_rf_model
from models.svm_model import get_svm_model
from models.logistic_model import get_logistic_model
from models.xgb_model import get_xgb_model 

# ----------------- Setup Configuration -----------------
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
voting_path = os.path.join(MODEL_DIR_MAIN, 'voting_clf.pkl')

# ----------------- Define Feature & Label Schema -----------------
# These lists ensure the model always sees features in the correct order.
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

diseases_list = [
    'Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
    'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
    'Migraine','Cervical spondylosis','Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox',
    'Dengue','Typhoid','hepatitis A','Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E',
    'Alcoholic hepatitis','Tuberculosis','Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
    'Heart attack','Varicose veins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
    'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
    'Impetigo'
]

# Remove accidental duplicates while preserving order
seen = set()
symptoms_list = [x for x in symptoms_list if not (x in seen or seen.add(x))]

# Save valid symptom list for user reference
with open("available_symptoms.txt", "w") as f:
    f.write("\n".join(sorted(symptoms_list)))

print(f"--- Configuration Loaded: {len(symptoms_list)} Symptoms, {len(diseases_list)} Diseases ---")

# ----------------- Data Loading & Preprocessing -----------------
try:
    df = pd.read_csv("Prototype.csv")
except FileNotFoundError:
    print("Error: 'Prototype.csv' not found. Please place it in the project directory.")
    exit()

# Prepare Labels
# We use LabelEncoder to ensure classes are 0, 1, 2... which helps XGBoost avoid errors
le = LabelEncoder()
df['prognosis'] = le.fit_transform(df['prognosis'])

# Ensure we remove any rows with missing labels
df.dropna(subset=['prognosis'], inplace=True)

X = df[symptoms_list]
y = df['prognosis'].astype(int)

# Update our disease lookup list to match the encoder's order
diseases_list = list(le.classes_)

# Split Data: 80% for training models, 20% for evaluating performance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Data Loaded: {len(X_train)} training samples, {len(X_test)} testing samples.")

# ----------------- Model Training -----------------
print("\n--- initializing Model Training ---")

models = {}
model_getters = {
    'KNN': get_knn_model,
    'Naive Bayes': get_nb_model,
    'Decision Tree': get_dt_model,
    'Random Forest': get_rf_model,
    'SVM': get_svm_model,
    'Logistic Regression': get_logistic_model,
    'XGBoost': get_xgb_model
}

# Train each model individually and save it
for name, getter_func in model_getters.items():
    path = model_paths[name]
    
    if os.path.exists(path):
        # Load existing model to save time
        models[name] = joblib.load(path)
    else:
        # Train new model using our helper functions
        print(f"Training {name}...")
        model = getter_func(X_train, y_train) 
        models[name] = model
        joblib.dump(model, path) 
    
    # Quick accuracy check
    y_pred = models[name].predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    print(f"{name} Accuracy: {acc:.2f}%")


# ----------------- Ensemble Learning (Voting) -----------------
print("\n--- initializing Voting Ensemble ---")

# We exclude XGBoost from the voting ensemble to keep it as a standalone benchmark
estimators = [(name, model) for name, model in models.items() if name != 'XGBoost']

# Weighted Voting Strategy:
# We give Random Forest a weight of 5 because it proved most robust in our testing.
# Order matches 'estimators': KNN, NB, DT, RF, SVM, LR
weights = [1, 1, 1, 5, 1, 1] 

if os.path.exists(voting_path):
    voting_clf = joblib.load(voting_path)
    print("Loaded existing Voting Classifier.")
else:
    print("Training Weighted Voting Classifier...")
    voting_clf = VotingClassifier(estimators=estimators, voting='soft', weights=weights)
    voting_clf.fit(X_train, y_train)
    joblib.dump(voting_clf, voting_path)

# Evaluate Ensemble
y_pred = voting_clf.predict(X_test)
print(f"Ensemble Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")


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
    # Get probability scores for all diseases
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
    print("\n===============================================")
    print("      AI Disease Prediction System v1.0")
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