import pandas as pd

def load_data(train_path="Prototype-1.csv", test_path="Prototype.csv", symptoms_list=[], diseases_list=[]):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Map disease names to numeric labels
    disease_mapping = {disease: i for i, disease in enumerate(diseases_list)}
    df_train['prognosis'] = df_train['prognosis'].map(disease_mapping)
    df_test['prognosis'] = df_test['prognosis'].map(disease_mapping)

    # Drop rows with unmapped diseases
    df_train.dropna(subset=['prognosis'], inplace=True)
    df_test.dropna(subset=['prognosis'], inplace=True)

    # Features and labels
    X_train = df_train[symptoms_list]
    y_train = df_train['prognosis'].astype(int)
    X_test = df_test[symptoms_list]
    y_test = df_test['prognosis'].astype(int)

    return X_train, X_test, y_train, y_test
