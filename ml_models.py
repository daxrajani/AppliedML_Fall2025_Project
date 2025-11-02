import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ML Models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# -----------------------------
# 1. Symptoms and Diseases
# -----------------------------
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

# ----------------- Load Data -----------------
df_train = pd.read_csv("Prototype-1.csv")
df_test = pd.read_csv("Prototype.csv")

# Map disease names to numeric labels
disease_mapping = {disease: i for i, disease in enumerate(diseases_list)}
df_train['prognosis'] = df_train['prognosis'].map(disease_mapping)
df_test['prognosis'] = df_test['prognosis'].map(disease_mapping)

# Drop rows with unmapped diseases (NaN labels)
df_train.dropna(subset=['prognosis'], inplace=True)
df_test.dropna(subset=['prognosis'], inplace=True)

# Features and labels
X_train = df_train[symptoms_list]
y_train = df_train['prognosis'].astype(int)

X_test = df_test[symptoms_list]
y_test = df_test['prognosis'].astype(int)

# ----------------- Train ML Models -----------------
models = {
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Extra Trees': ExtraTreesClassifier(),
    'SVM': SVC(probability=True)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    print(f"{name} Accuracy: {acc:.2f}%")

# ----------------- Soft Voting -----------------
estimators = [(name, model) for name, model in models.items()]
voting_clf = VotingClassifier(estimators=estimators, voting='soft')
voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)
print(f"Voting Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# ----------------- Predict from User Input -----------------
def predict_disease(symptoms_input):
    input_vector = [1 if symptom in symptoms_input else 0 for symptom in symptoms_list]
    input_df = pd.DataFrame([input_vector], columns=symptoms_list)

    # Predict using Voting classifier
    pred_label = voting_clf.predict(input_df)[0]
    predicted_disease = diseases_list[pred_label]
    return predicted_disease

# ----------------- Main -----------------
if __name__ == "__main__":
    user_input = input("Enter 5 symptoms separated by comma: ")
    user_symptoms = [s.strip() for s in user_input.split(",")]
    predicted_disease = predict_disease(user_symptoms)
    print(f"Predicted Disease (using soft voting): {predicted_disease}")