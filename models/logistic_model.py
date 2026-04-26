from sklearn.linear_model import LogisticRegression

def get_logistic_model(X_train, y_train):
    
    # --- Parameters found by your tuning script ---
    print("Training LR")

    model = LogisticRegression(
        C=0.1,
        penalty='l2',
        solver='lbfgs',
        random_state=42,
        max_iter=2000
    )
    
    model.fit(X_train, y_train)
    return model