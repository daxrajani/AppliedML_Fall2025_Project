from sklearn.linear_model import LogisticRegression

def get_logistic_model(X_train, y_train):
    model = LogisticRegression(
        C=0.01,
        solver='lbfgs',
        max_iter=100,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model
