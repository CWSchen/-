from minepy import MINE
import numpy as np

m = MINE()
x = np.random.uniform(-1, 1, 10000)
m.compute_score(x, x ** 2)
print(m.mic())  # 1.0000000000000009

import numpy as np
from scipy.stats import pearsonr

np.random.seed(0)
size = 300
x = np.random.normal(0, 1, size)
print("Lower noise", pearsonr(x, x + np.random.normal(0, 1, size)))
print("Higher noise", pearsonr(x, x + np.random.normal(0, 10, size)))
'''
Lower noise (0.7182483686213841, 7.32401731299835e-49)
Higher noise (0.057964292079338155, 0.3170099388532475)
'''
