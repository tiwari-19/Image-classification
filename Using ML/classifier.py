import os
import _pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


np.random.seed(7)
CHECKPOINT_PATH = os.path.join(os.getcwd(), 'checkpoint_dir')
data = _pickle.load(open(os.path.join(CHECKPOINT_PATH, 'dataset_feats.pkl'), 'rb'))
Labels = _pickle.load(open(os.path.join(CHECKPOINT_PATH, 'labels.pkl'), 'rb'))

X_train, X_val, y_train, y_val = train_test_split(data, Labels, test_size=0.10, stratify=Labels)
print("Training samples =", X_train.shape[0])
print("Validation samples =", X_val.shape[0])

clf = RandomForestClassifier(n_estimators=200)
clf.fit(X_train, y_train)
print(accuracy_score(clf.predict(X_val), y_val))

_pickle.dump(clf, open(os.path.join(CHECKPOINT_PATH, 'classifier.pkl'), 'wb'))
