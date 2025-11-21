from sklearn.ensemble import RandomForestClassifier

def get_rf_model(X_train, y_train):
    
    # --- Parameters found by your tuning script ---
    print("Training RF")
    
    model = RandomForestClassifier(
        n_estimators=100,
        min_samples_split=2,
        min_samples_leaf=4,
        max_depth=20,
        criterion='gini',
        random_state=42
    ) 
    
    model.fit(X_train, y_train)
    return model