from sklearn.tree import DecisionTreeClassifier

def get_dt_model(X_train, y_train):
    
    # --- Parameters found by your tuning script ---
    print("Training DT")

    model = DecisionTreeClassifier(
        min_samples_split=2,
        min_samples_leaf=2,
        max_depth=30,
        criterion='gini',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    return model