from sklearn.tree import DecisionTreeClassifier

def get_dt_model(X_train, y_train):
    
  
    print("Training Decision Tree")

    model = DecisionTreeClassifier(
        criterion='entropy',      # Changed from 'gini'
        max_depth=20,             # Changed from 30
        min_samples_split=2,      # Kept as 2
        min_samples_leaf=1,       # Reset to default (1) because your tuning script did not include this parameter
        random_state=42
    )
    
    model.fit(X_train, y_train)
    return model