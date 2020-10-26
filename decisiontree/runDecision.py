import numpy as np
from arff import Arff
from decisiontree import DTClassifier
from sklearn.linear_model import Perceptron

filepath = "test.arff"  # example
# filepath = 'lenses.arff'
# filepath = 'evaluation.arff'
# filepath = 'car.arff'
# filepath = 'vote.arff'
af = Arff()
af.load_arff(filepath)
all_data = af.data
for i in range(len(all_data)):  # if data contains nan, change nan to -1
    for j in range(len(all_data[i])):
        if np.isnan(all_data[i][j]):
            all_data[i][j] = -1
shuffle = True
split = True
if shuffle:
    np.random.shuffle(all_data)
if split:
    split = int(round(.75 * len(all_data)))
    print(split)
    trainingset = all_data[:split]
    testingset = all_data[split:]
else:
    trainingset = all_data
    testingset = all_data

dt = DTClassifier()
dt.fit(trainingset[:, :-1], trainingset[:, -1])
print(dt.score(all_data[:, :-1], all_data[:, -1]))
print(dt.score(testingset[:, :-1], testingset[:, -1]))
