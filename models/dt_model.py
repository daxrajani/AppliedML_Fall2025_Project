from sklearn.tree import DecisionTreeClassifier

def get_dt_model(X_train, y_train):
    model = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1
    )
    model.fit(X_train, y_train)
    return model
