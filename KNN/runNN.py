import numpy as np
from arff import Arff
from KNN import KNNClassifier


filepath = 'diabetes.arff'
af = Arff()
af.load_arff(filepath)
all_data = af.data
catagory = np.zeros([len(all_data[0])])
weight_type=['inverse_distance','no_weight']

KK = KNNClassifier(catagory, k=3, weight_type=weight_type[1])

KK.fit(all_data[:, :-1], all_data[:, -1])

filepath = 'diabetes_test.arff'
af.load_arff(filepath)
test = af.data

print(KK.score(test[:, :-1], test[:, -1]))
