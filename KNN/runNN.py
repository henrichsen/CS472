import numpy as np
from arff import Arff
from KNN import KNNClassifier


filepath = 'step5_test.arff'
af = Arff()
af.load_arff(filepath)
all_data = af.data
catagory = np.zeros([len(all_data[0])])
weight_type=['inverse_distance','no_weight']

KK = KNNClassifier(catagory, k=-1, weight_type=weight_type[0], normalize=True)

KK.fit(all_data[:, :-1], all_data[:, -1])

filepath = 'step5_train.arff'
af.load_arff(filepath)
test = af.data

print(KK.score(test[:, :-1], test[:, -1]))
