from xgboost import XGBClassifier

def get_xgb_model(X_train, y_train):
    print("Training XGBoost")
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.7,
        colsample_bytree=0.7,
        eval_metric='mlogloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model