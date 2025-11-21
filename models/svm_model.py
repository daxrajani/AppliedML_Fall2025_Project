from sklearn.svm import SVC

def get_svm_model(X_train, y_train):
    
    # --- Parameters found by your tuning script ---
    print("Training SVM")
    
    model = SVC(
        probability=True,  # Required for Soft Voting
        kernel='rbf',
        gamma=0.001,
        C=1,
        random_state=42
    ) 
    
    model.fit(X_train, y_train)
    return model