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
# Get the directory where this script (evaluation.py) is located: .../Model_development/scripts
script_dir = os.path.dirname(os.path.abspath(__file__))

# Go UP one level to get the main project folder: .../Model_development
project_root = os.path.dirname(script_dir)

# Define paths relative to the project root
MODEL_DIR = os.path.join(project_root, "saved_models_main")
OUTPUT_DIR = os.path.join(project_root, "evaluation_results")
csv_path = os.path.join(project_root, "Prototype.csv") 

# Create output directories in the main project folder if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "confusion_matrices"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "interpretability"), exist_ok=True)

print(f"Project Root detected as: {project_root}")
print(f"Looking for models in: {MODEL_DIR}")
print(f"Saving results to: {OUTPUT_DIR}")

# ----------------- Data Loading (MATCHES MAIN.PY EXACTLY) -----------------
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found. Make sure it is in the main project folder.")
    exit()

print(f"Loading data from {csv_path}...")
df = pd.read_csv(csv_path)

# --- Define Feature & Label Schema (Copied from main.py) ---
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

# Remove accidental duplicates while preserving order
seen = set()
symptoms_list = [x for x in symptoms_list if not (x in seen or seen.add(x))]

# Prepare Labels using LabelEncoder
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

print(f"Data Loaded: {len(X_test)} test samples.")

# ----------------- Load Models -----------------
model_paths = {
    'KNN': 'knn.pkl', 
    'Naive Bayes': 'nb.pkl', 
    'Decision Tree': 'dt.pkl',
    'Random Forest': 'rf.pkl', 
    'SVM': 'svm.pkl', 
    'Logistic Regression': 'lr.pkl',
    'XGBoost': 'xgb.pkl', 
    'Voting Classifier': 'voting_clf.pkl'
}

models = {}
for name, file in model_paths.items():
    path = os.path.join(MODEL_DIR, file)
    if os.path.exists(path): 
        models[name] = joblib.load(path)
    else:
        print(f"Warning: Model file not found at {path}")

if not models: 
    exit("No models found. Please run 'main.py' in the root folder first.")

# ----------------- 1. Quantitative Evaluation -----------------
print("\n--- Computing Metrics ---")
all_metrics = {}
all_predictions = {}
expected_labels = list(range(len(diseases_list)))

best_f1 = 0
best_model_name = ""
top_confusion_str = "None"

for name, model in models.items():
    print(f"Evaluating: {name}")
    y_pred = model.predict(X_test)
    all_predictions[name] = y_pred
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name

    # Specificity (Handling multi-class by averaging One-vs-Rest)
    cm = confusion_matrix(y_test, y_pred, labels=expected_labels)
    fp = cm.sum(axis=0) - np.diag(cm)
    tn = cm.sum() - (fp + cm.sum(axis=1) - np.diag(cm) + np.diag(cm))
    with np.errstate(divide='ignore', invalid='ignore'):
        spec = np.mean(np.nan_to_num(tn / (tn + fp)))

    all_metrics[name] = {
        'Accuracy': acc, 'Precision': prec, 'Recall': rec, 
        'F1-Score': f1, 'Specificity': spec, 'MCC': mcc
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

# Save Metrics Table
metrics_df = pd.DataFrame(all_metrics).T
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "comparative_metrics.csv"))
print("Metrics saved.")

# ----------------- 2. Qualitative Evaluation -----------------
print("\n--- Generating Reports ---")

# Error Analysis
error_path = os.path.join(OUTPUT_DIR, "error_analysis.txt")
with open(error_path, 'w') as f:
    model_name = 'Voting Classifier' if 'Voting Classifier' in all_predictions else list(all_predictions.keys())[0]
    y_p = all_predictions[model_name]
    
    # Identify indices where prediction matches mismatch
    misses = np.where(y_test.values != y_p)[0]
    
    f.write(f"Model Analyzed: {model_name}\nTotal Misclassified: {len(misses)}\n\n")
    
    f.write("--- Top Confused Pairs ---\n")
    cm = confusion_matrix(y_test, y_p)
    np.fill_diagonal(cm, 0)
    
    if cm.sum() > 0:
        params = np.unravel_index(np.argsort(cm, axis=None)[-3:], cm.shape)
        for i in range(len(params[0])-1, -1, -1):
            true_idx, pred_idx = params[0][i], params[1][i]
            count = cm[true_idx, pred_idx]
            if count > 0:
                msg = f"{count} times: True='{diseases_list[true_idx]}' -> Pred='{diseases_list[pred_idx]}'\n"
                f.write(msg)
                if top_confusion_str == "None": top_confusion_str = f"{diseases_list[true_idx]} vs {diseases_list[pred_idx]}"
    
    f.write("\n--- Detailed Cases ---\n")
    # Iterate through first 5 misses to give examples
    for i in misses[:5]:
        true_label_name = diseases_list[y_test.iloc[i]]
        pred_label_name = diseases_list[y_p[i]]
        f.write(f"True: {true_label_name} -> Pred: {pred_label_name}\n")
        
        # Get active symptoms for this case
        row_data = X_test.iloc[i]
        active_symptoms = list(row_data[row_data == 1].index)
        f.write(f"Symptoms: {active_symptoms}\n\n")

# SHAP Analysis
try:
    rf = models.get('Random Forest')
    if rf:
        print("Generating SHAP...")
        # Sample a small subset of test data for visualization
        X_s = X_test.sample(n=min(100, len(X_test)), random_state=42)
        
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_s, check_additivity=False)
        
        # Handle binary vs multi-class shape issues
        vals = np.abs(shap_values[0]) if isinstance(shap_values, list) else np.abs(shap_values)
        if isinstance(shap_values, list):
             for k in range(1, len(shap_values)): vals += np.abs(shap_values[k])
        
        shap.summary_plot(vals, X_s, plot_type='bar', show=False)
        plt.savefig(os.path.join(OUTPUT_DIR, "interpretability", "shap_summary.png"))
        plt.close()
except Exception as e: print(f"SHAP Error: {e}")

# LIME Analysis
try:
    v_clf = models.get('Voting Classifier')
    if v_clf and len(misses) > 0:
        print("Generating LIME...")
        # Pick the first misclassified instance
        idx_loc = misses[0]
        
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train.values, feature_names=symptoms_list, class_names=diseases_list, mode='classification'
        )
        
        # Explain the prediction
        exp = explainer.explain_instance(X_test.iloc[idx_loc].values, v_clf.predict_proba, num_features=10, top_labels=3)
        exp.save_to_file(os.path.join(OUTPUT_DIR, "interpretability", f"lime_case_{idx_loc}.html"))
except Exception as e: print(f"LIME Error: {e}")

# ----------------- 3. Final Summary & Insights -----------------
print("\n" + "="*60)
print("📊  FINAL EVALUATION SUMMARY")
print("="*60)
print(metrics_df.to_string(float_format="%.4f"))
print("-" * 60)
print(f"🏆 Best Model:        {best_model_name} (F1-Score: {best_f1:.4f})")
print(f"🔄 Top Confusion:     {top_confusion_str}")
print("="*60 + "\n")
print(f"Results saved in: {OUTPUT_DIR}")