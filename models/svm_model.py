from sklearn.svm import SVC

def get_svm_model(X_train, y_train):
    model = SVC(
        C=0.1,
        gamma=1,
        kernel='rbf',
        probability=True,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model
