from sklearn.naive_bayes import GaussianNB
import numpy as np

def get_nb_model(X_train, y_train):
    
    # --- Parameters found by your tuning script ---
    # Note: Your tuning found 1.0 is the best value.
    print("Training NB")

    model = GaussianNB(
        var_smoothing=1.0
    )
    
    model.fit(X_train, y_train)
    return model