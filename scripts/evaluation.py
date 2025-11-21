import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_score, recall_score, f1_score, matthews_corrcoef
)
from sklearn.preprocessing import LabelEncoder
import shap
import lime
import lime.lime_tabular

# ----------------- Setup Directories -----------------
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
MODEL_DIR = os.path.join(base_dir, "saved_models_main")
OUTPUT_DIR = os.path.join(base_dir, "evaluation_results")
csv_path = os.path.join(base_dir, "Prototype.csv") 

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "confusion_matrices"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "interpretability"), exist_ok=True)

# ----------------- Data Loading (MATCHES MAIN.PY) -----------------
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found.")
    exit()

print(f"Loading data from {csv_path}...")
df = pd.read_csv(csv_path)

# --- Hardcoded Schema ---
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

# Prepare Data
mapping = {d: i for i, d in enumerate(diseases_list)}
df['prognosis'] = df['prognosis'].map(mapping)
df.dropna(subset=['prognosis'], inplace=True)

# Clean duplicates
seen = set()
symptoms_list = [x for x in symptoms_list if not (x in seen or seen.add(x))]

X = df[symptoms_list]
y = df['prognosis'].astype(int)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Data Loaded: {len(X_test)} test samples.")

# ----------------- Load Models -----------------
print(f"Loading models from: {MODEL_DIR}")
model_paths = {
    'KNN': 'knn.pkl', 'Naive Bayes': 'nb.pkl', 'Decision Tree': 'dt.pkl',
    'Random Forest': 'rf.pkl', 'SVM': 'svm.pkl', 'Logistic Regression': 'lr.pkl',
    'XGBoost': 'xgb.pkl', 'Voting Classifier': 'voting_clf.pkl'
}

models = {}
for name, file in model_paths.items():
    path = os.path.join(MODEL_DIR, file)
    if os.path.exists(path): 
        models[name] = joblib.load(path)

if not models: 
    exit("No models found. Run main.py first.")

# ----------------- 1. Quantitative Evaluation -----------------
print("\n--- Computing Metrics ---")
all_metrics = {}
all_predictions = {}
expected_labels = list(range(len(diseases_list)))

for name, model in models.items():
    print(f"Evaluating: {name}")
    y_pred = model.predict(X_test)
    all_predictions[name] = y_pred
    
    # Advanced Metrics
    mcc = matthews_corrcoef(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=expected_labels)
    fp = cm.sum(axis=0) - np.diag(cm)
    tn = cm.sum() - (fp + cm.sum(axis=1) - np.diag(cm) + np.diag(cm))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        spec = np.mean(np.nan_to_num(tn / (tn + fp)))

    all_metrics[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'Specificity': spec, 
        'MCC': mcc
    }
    
    # Plots
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=diseases_list, yticklabels=diseases_list)
    plt.title(f'Confusion Matrix: {name}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrices", f"cm_{name.replace(' ', '_')}.png"))
    plt.close()

    with open(os.path.join(OUTPUT_DIR, f"report_{name.replace(' ', '_')}.txt"), 'w') as f:
        f.write(classification_report(y_test, y_pred, labels=expected_labels, target_names=diseases_list, zero_division=0))

pd.DataFrame(all_metrics).T.to_csv(os.path.join(OUTPUT_DIR, "comparative_metrics.csv"))
print("Metrics saved.")

# ----------------- 2. Qualitative Evaluation -----------------
print("\n--- Generating Reports ---")

# A. Error Analysis (Hard Cases)
error_path = os.path.join(OUTPUT_DIR, "error_analysis.txt")
with open(error_path, 'w') as f:
    model_name = 'Voting Classifier' if 'Voting Classifier' in all_predictions else list(all_predictions.keys())[0]
    y_p = all_predictions[model_name]
    misses = np.where(y_test != y_p)[0]
    
    f.write(f"Model Analyzed: {model_name}\nTotal Misclassified: {len(misses)}\n\n")
    
    # --- NEW: Most Confused Pairs ---
    f.write("--- Top Confused Pairs ---\n")
    cm = confusion_matrix(y_test, y_p)
    np.fill_diagonal(cm, 0) # Ignore correct predictions
    # Find indices of top 3 errors
    params = np.unravel_index(np.argsort(cm, axis=None)[-3:], cm.shape)
    for i in range(len(params[0])-1, -1, -1): # Iterate backwards (highest first)
        true_idx, pred_idx = params[0][i], params[1][i]
        count = cm[true_idx, pred_idx]
        if count > 0:
            f.write(f"{count} times: True='{diseases_list[true_idx]}' -> Pred='{diseases_list[pred_idx]}'\n")
    
    f.write("\n--- Detailed Cases ---\n")
    for i in misses[:5]:
        f.write(f"True: {diseases_list[y_test.iloc[i]]} -> Pred: {diseases_list[y_p[i]]}\n")
        f.write(f"Symptoms: {list(X_test.iloc[i][X_test.iloc[i]==1].index)}\n\n")

# B. SHAP (Interpretability)
try:
    rf = models.get('Random Forest')
    if rf:
        print("Generating SHAP...")
        X_s = X_test.sample(n=min(100, len(X_test)), random_state=42)
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_s, check_additivity=False)
        vals = np.abs(shap_values[0]) if isinstance(shap_values, list) else np.abs(shap_values)
        if isinstance(shap_values, list):
             for k in range(1, len(shap_values)): vals += np.abs(shap_values[k])
        shap.summary_plot(vals, X_s, plot_type='bar', show=False)
        plt.savefig(os.path.join(OUTPUT_DIR, "interpretability", "shap_summary.png"))
        plt.close()
except Exception as e: print(f"SHAP Error: {e}")

# C. LIME (Local Explanations)
try:
    v_clf = models.get('Voting Classifier')
    if v_clf and len(misses) > 0:
        print("Generating LIME...")
        idx = misses[0]
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train.values, feature_names=symptoms_list, class_names=diseases_list, mode='classification'
        )
        exp = explainer.explain_instance(X_test.iloc[idx].values, v_clf.predict_proba, num_features=10, top_labels=3)
        exp.save_to_file(os.path.join(OUTPUT_DIR, "interpretability", f"lime_case_{idx}.html"))
except Exception as e: print(f"LIME Error: {e}")

print(f"\nDone! Results in {OUTPUT_DIR}")