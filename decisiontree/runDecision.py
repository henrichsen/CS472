import numpy as np
from arff import Arff
from decisiontree import DTClassifier
from sklearn import tree
import graphviz

dt = DTClassifier()
# filepath = "test.arff"  # example
# filepath = 'lenses.arff'
#filepath = 'evaluation.arff'
filepath = 'car.arff'
#filepath = 'vote.arff'
af = Arff()
af.load_arff(filepath)
all_data = af.data
for i in range(len(all_data)):  # if data contains nan, change nan to -1
    for j in range(len(all_data[i])):
        if np.isnan(all_data[i][j]):
            all_data[i][j] = -1
shuffle = True
split = False
if shuffle:
    np.random.shuffle(all_data)
if split:
    sum = 0
    cv_array = np.array_split(all_data, 10.0, 0)
    for j in range(len(cv_array)):
        testset = cv_array[j]
        trainingset = np.concatenate([cv_array[i] for i in range(len(cv_array)) if i != j])
        dt.fit(trainingset[:, :-1], trainingset[:, -1])
        sum += dt.score(testset[:, :-1], testset[:, -1])
        print(dt.score(testset[:, :-1], testset[:, -1]))
    print('Average accuracy: ' + str(sum / 10))
else:
    sp=int(.75*len(all_data))
    trainingset = all_data[:sp]
    testingset = all_data[sp:]
    dt.fit(trainingset[:, :-1], trainingset[:, -1])
    print(dt.score(all_data[:, :-1], all_data[:, -1]))
    print(dt.score(testingset[:, :-1], testingset[:, -1]))
    print('__________________________')
    clf = tree.DecisionTreeClassifier(max_features=1)
    clf = clf.fit(trainingset[:, :-1], trainingset[:, -1])
    print(clf.score(testingset[:, :-1], testingset[:, -1]))
