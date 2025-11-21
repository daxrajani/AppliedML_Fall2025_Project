import pandas as pd
import numpy as np
import os # Import os to manage paths

# --- Re-using the cleaning functions ---

def clean_header(header):
    """Cleans 'UMLS:C0018681_headache' -> 'headache'"""
    if header.startswith('UMLS:'):
        parts = header.split('_', 1)
        if len(parts) > 1:
            name = parts[1].replace('"', '').strip()
            if name.endswith(', natural'):
                name = name.replace(', natural', '')
            return name
    if header == 'label':
        return 'prognosis'
    return header

def clean_label(label_str):
    """Cleans 'UMLS:C0011570_depression mental^...' -> 'depression mental'"""
    try:
        first_label = str(label_str).split('^')[0]
        name = first_label.split('_', 1)[1]
        return name
    except Exception:
        return label_str

# --- Main Generation Script ---

SAMPLES_PER_DISEASE = 120

# --- Paths logic (looks for main.csv in the same folder) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
TEMPLATE_FILE = os.path.join(script_dir, "main.csv")
NEW_PROTOTYPE_FILE = os.path.join(base_dir, "Prototype_GENERATED.csv")

print(f"Loading template file: {TEMPLATE_FILE}...")
try:
    df_template = pd.read_csv(TEMPLATE_FILE, on_bad_lines='skip')
except FileNotFoundError:
    print(f"Error: '{TEMPLATE_FILE}' not found.")
    print(f"Please ensure 'main.csv' is in the same folder as this script:")
    print(f"{script_dir}")
    exit()

# 1. Clean the template file's headers
print("Cleaning template file headers...")
df_template.columns = [clean_header(col) for col in df_template.columns]

# 2. Clean the 'prognosis' (label) column
if 'prognosis' in df_template.columns:
    print("Cleaning disease labels...")
    df_template['prognosis'] = df_template['prognosis'].apply(clean_label)
else:
    print("Error: 'label' column not found. Cannot proceed.")
    exit()

# 3. Drop the 'frequency' column
if 'frequency' in df_template.columns:
    df_template = df_template.drop(columns=['frequency'])

# 4. Get the list of symptom columns
symptom_columns = [col for col in df_template.columns if col != 'prognosis']

# 5. Generate the new dataset
print(f"Generating {SAMPLES_PER_DISEASE} samples for each disease...")
all_generated_rows = [] # This will be a list of dictionaries

# Loop through each row in the template file (each disease)
for _, rule_row in df_template.iterrows():
    
    disease_name = rule_row['prognosis']
    
    # Get the template of symptoms for this disease
    symptom_template = rule_row[symptom_columns]
    
    # Create 120 copies for this one disease
    for _ in range(SAMPLES_PER_DISEASE):
        new_row_dict = symptom_template.to_dict()
        new_row_dict['prognosis'] = disease_name
        all_generated_rows.append(new_row_dict)

# 6. Create the final DataFrame
print("Assembling final DataFrame...")
df_generated = pd.DataFrame(all_generated_rows)

# 7. Merge duplicate columns (FIXED LINE)
print("Merging duplicate symptom columns (e.g., 'underweight')...")
# This is the new, modern way to do this and will not produce a warning
df_generated = df_generated.T.groupby(level=0).max().T

# 8. Re-order columns to put 'prognosis' at the end
print("Finalizing file...")
if 'prognosis' in df_generated.columns:
    prognosis_col = df_generated.pop('prognosis')
    df_generated['prognosis'] = prognosis_col
else:
    print("Warning: 'prognosis' column was lost during generation.")

# 9. Save the new file
df_generated.to_csv(NEW_PROTOTYPE_FILE, index=False)

print(f"\nSuccess! New file created: {NEW_PROTOTYPE_FILE}")
print(f"(Saved in your main project folder: {base_dir})")
print(f"Total diseases: {len(df_template)}")
print(f"Samples per disease: {SAMPLES_PER_DISEASE}")
print(f"Total rows generated: {len(df_generated)}")
print(f"Total unique symptoms (columns): {len(df_generated.columns) - 1}")