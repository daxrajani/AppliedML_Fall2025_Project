from sklearn.ensemble import RandomForestClassifier

def get_rf_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        criterion='entropy',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model
