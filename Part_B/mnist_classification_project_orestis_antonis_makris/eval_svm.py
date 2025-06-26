from sklearn.metrics import confusion_matrix, accuracy_score
from hog_utils import extract_hog_dataset
import joblib

if __name__=='__main__':
    svc = joblib.load('svm_mnist.pkl')
    X_test, y_test = extract_hog_dataset(train=False)
    y_pred = svc.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(f"SVM Accuracy: {acc:.4f}")
    print(cm)