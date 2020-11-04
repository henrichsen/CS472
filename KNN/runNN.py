import numpy as np
from arff import Arff
from KNN import KNNClassifier

filepath = 'diabetes.arff'
af = Arff()
af.load_arff(filepath)
all_data = af.data
#0, -1, -1, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1 - 1
catagory = np.zeros(len(all_data))
#catagory[1] = -1
#catagory[2] = -1
#catagory[7] = -1
#catagory[10] = -1
#catagory[13] = -1
#catagory[14] = -1
weight_type = ['inverse_distance', 'no_weight']

KK = KNNClassifier(catagory, k=15, weight_type=weight_type[0], normalize=True)

KK.fit(all_data[:, :-1], all_data[:, -1])

print('fitted')

filepath = 'diabetes_test.arff'
af.load_arff(filepath)
test = af.data
print(KK.score(test[:, :-1], test[:, -1]))