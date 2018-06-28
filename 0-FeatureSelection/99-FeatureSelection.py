from sklearn.datasets import load_boston
from sklearn.linear_model import (LinearRegression, Ridge,
                                  Lasso, RandomizedLasso)
from sklearn.feature_selection import RFE, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from minepy import MINE
from pandas import DataFrame

np.random.seed(0)
size = 750
X = np.random.uniform(0, 1, (size, 14))
# "Friedamn #1” regression problem
Y = (10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - .5) ** 2 +
     10 * X[:, 3] + 5 * X[:, 4] + np.random.normal(0, 1))
# Add 3 additional correlated variables (correlated with X1-X3)
X[:, 10:] = X[:, :4] + np.random.normal(0, .025, (size, 4))
names = ["x%s" % i for i in range(1, 15)]
ranks = {}


def rank_to_dict(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order * np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks))


lr = LinearRegression(normalize=True)
lr.fit(X, Y)
ranks["Linear reg"] = rank_to_dict(np.abs(lr.coef_), names)
ridge = Ridge(alpha=7)
ridge.fit(X, Y)
ranks["Ridge"] = rank_to_dict(np.abs(ridge.coef_), names)
lasso = Lasso(alpha=.05)
lasso.fit(X, Y)
ranks["Lasso"] = rank_to_dict(np.abs(lasso.coef_), names)
rlasso = RandomizedLasso(alpha=0.04)
rlasso.fit(X, Y)
ranks["Stability"] = rank_to_dict(np.abs(rlasso.scores_), names)
#stop the search when 5 features are left (they will get equal scores)
rfe = RFE(lr, n_features_to_select=5)
rfe.fit(X, Y)

'''
ranks["RFE"] = rank_to_dict(map(float, rfe.ranking_), names, order=-1)
python 3 map 需要转 list
'''
print('  ', list(map(float, rfe.ranking_)))
# [1.0, 1.0, 1.0, 1.0, 3.0, 6.0, 10.0, 5.0, 9.0, 7.0, 1.0, 4.0, 2.0, 8.0]

ranks["RFE"] = rank_to_dict(list(map(float, rfe.ranking_)), names, order=-1)
rf = RandomForestRegressor()
rf.fit(X, Y)
ranks["RF"] = rank_to_dict(rf.feature_importances_, names)
f, pval = f_regression(X, Y, center=True)
ranks["Corr."] = rank_to_dict(f, names)
mine = MINE()
mic_scores = []
for i in range(X.shape[1]):
    mine.compute_score(X[:, i], Y)
    m = mine.mic()
    mic_scores.append(m)
ranks["MIC"] = rank_to_dict(mic_scores, names)
r = {}
for name in names:
    r[name] = round(np.mean([ranks[method][name]
                             for method in ranks.keys()]), 2)
methods = sorted(ranks.keys())
ranks["Mean"] = r
methods.append("Mean")
print("\t%s" % "\t".join(methods))

for name in names:
    print("%s\t%s" % (name, "\t".join(map(str, [ranks[method][name] for method in methods]))))

print(methods)
print(names)
print(['x%s' % x for x in range(1, len(names) + 1)])
'''
['Corr.', 'Lasso', 'Linear reg', 'MIC', 'RF', 'RFE', 'Ridge', 'Stability', 'Mean']
['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14']
['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14']
'''
print(ranks)

arr_ = []
for name in names:
    arr_.append(list(map(str, [ranks[method][name] for method in methods])))
print(arr_)

print('-' * 100)
data = DataFrame(arr_, index=names, columns=methods)
print(data)
'''
    Corr. Lasso Linear reg   MIC    RF   RFE Ridge Stability  Mean
x1    0.3  0.79        1.0  0.39  0.55   1.0  0.77      0.77   0.7
x2   0.44  0.83       0.56  0.61  0.67   1.0  0.75      0.72   0.7
x3    0.0   0.0        0.5  0.34  0.13   1.0  0.05       0.0  0.25
x4    1.0   1.0       0.57   1.0  0.56   1.0   1.0       1.0  0.89
x5    0.1  0.51       0.27   0.2  0.29  0.78  0.88      0.55  0.45
x6    0.0   0.0       0.02   0.0  0.01  0.44  0.05       0.0  0.06
x7   0.01   0.0        0.0  0.07  0.02   0.0  0.01       0.0  0.01
x8   0.02   0.0       0.03  0.05  0.01  0.56  0.09       0.0   0.1
x9   0.01   0.0        0.0  0.09  0.01  0.11   0.0       0.0  0.03
x10   0.0   0.0       0.01  0.04   0.0  0.33  0.01       0.0  0.05
x11  0.29   0.0        0.6  0.43  0.39   1.0  0.59      0.37  0.46
x12  0.44   0.0       0.14  0.71  0.35  0.67  0.68      0.46  0.43
x13   0.0   0.0       0.48  0.23  0.07  0.89  0.02       0.0  0.21
x14  0.99  0.16        0.0   1.0   1.0  0.22  0.95      0.62  0.62
'''
