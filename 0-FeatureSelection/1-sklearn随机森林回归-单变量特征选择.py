from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
import numpy as np

#Load boston housing dataset as an example
boston = load_boston()
X = boston["data"]
Y = boston["target"]
names = boston["feature_names"]

rf = RandomForestRegressor(n_estimators=20, max_depth=4)
scores = []
for i in range(X.shape[1]):
     score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2",
                              cv=ShuffleSplit(len(X), 3, .3))
     scores.append((round(np.mean(score), 3), names[i]))
print (sorted(scores, reverse=True))
'''
[(0.655, 'LSTAT'), (0.587, 'RM'), (0.418, 'NOX'), (0.387, 'TAX'),
 (0.31, 'INDUS'), (0.285, 'PTRATIO'), (0.223, 'ZN'), (0.206, 'CRIM'),
 (0.197, 'RAD'), (0.111, 'AGE'), (0.087, 'B'), (0.064, 'DIS'), (-0.006, 'CHAS')]
'''