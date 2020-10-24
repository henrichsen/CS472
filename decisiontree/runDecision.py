import numpy as np
from arff import Arff
from decisiontree import DTClassifier
from sklearn.linear_model import Perceptron

filepath = "test.arff"  # example
filepath = 'car.arff'
filepath = 'vote.arff'
af = Arff()
af.load_arff(filepath)
all_data = af.data
for i in range(len(all_data)):
    for j in range(len(all_data[i])):
        if np.isnan(all_data[i][j]):
            all_data[i][j] = -1

print(all_data)

dt = DTClassifier()
dt.fit(all_data[:, :-1], all_data[:, -1])
print(dt.score(all_data[:, :-1], all_data[:, -1]))
