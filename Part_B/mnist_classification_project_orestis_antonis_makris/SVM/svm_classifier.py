from sklearn.svm import SVC

def train_svm(X_train, y_train, C=1.0, kernel='linear'):
    """
    Εκπαίδευση Ssvm.
    """
    model = SVC(C=C, kernel=kernel)
    model.fit(X_train, y_train)
    return model

def predict_svm(model, X):
    
    return model.predict(X)