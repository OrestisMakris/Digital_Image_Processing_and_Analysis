from sklearn.svm import SVC
from hog_utils import extract_hog_dataset
import joblib

if __name__=='__main__':
    X, y = extract_hog_dataset(train=True)
    svc = SVC(kernel='linear')
    svc.fit(X, y)
    joblib.dump(svc, 'svm_mnist.pkl')