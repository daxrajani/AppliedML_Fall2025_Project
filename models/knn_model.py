from sklearn.neighbors import KNeighborsClassifier

def get_knn_model(X_train, y_train):
    
    # --- Parameters found by your tuning script ---
    print("Training KNN")
    
    model = KNeighborsClassifier(
        weights='distance',
        n_neighbors=11,
        metric='euclidean'
    ) 
    
    model.fit(X_train, y_train)
    return model