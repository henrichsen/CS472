import numpy as np
from arff import Arff
from KNN import KNNClassifier

filepath = 'Magic_train.arff'
af = Arff()
af.load_arff(filepath)
all_data = af.data
catagory = np.zeros([len(all_data[0])-1])
weight_type = ['inverse_distance', 'no_weight']

KK = KNNClassifier(catagory, k=3, weight_type=weight_type[1], normalize=True)

KK.fit(all_data[:, :-1], all_data[:, -1])

print('fitted')

filepath = 'Magic_test.arff'
af.load_arff(filepath)
test = af.data
print(KK.score(test[:, :-1], test[:, -1]))
