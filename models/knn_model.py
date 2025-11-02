from sklearn.neighbors import KNeighborsClassifier

def get_knn_model(X_train, y_train):
    model = KNeighborsClassifier(
        n_neighbors=3,
        weights='uniform',
        metric='euclidean'
    )
    model.fit(X_train, y_train)
    return model
