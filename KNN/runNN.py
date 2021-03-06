import numpy as np
from arff import Arff
from KNN import KNNClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

filepath = 'iris.arff'
af = Arff()
af.load_arff(filepath)
all_data = af.data
# filepath = 'regression_test.arff'
np.random.shuffle(all_data)
split = int(.7 * len(all_data))
test = all_data[split:,:]
all_data = all_data[:split,:]

KNR = KNeighborsRegressor(n_neighbors=7, weights='uniform', algorithm='ball_tree')
KNR.fit(all_data[:, :-1], all_data[:, -1])
print(KNR.score(test[:, :-1], test[:, -1]))

filepath = 'Magic_train.arff'
af = Arff()
af.load_arff(filepath)
all_data = af.data
filepath = 'Magic_test.arff'
af.load_arff(filepath)
test = af.data

KNC = KNeighborsClassifier(n_neighbors=3, weights='uniform', )
KNC.fit(all_data[:, :-1], all_data[:, -1])
print(KNC.score(test[:, :-1], test[:, -1]))

# 0, -1, -1, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1 - 1
catagory = np.zeros(len(all_data)) * -1
# catagory[3]= 0
# catagory[1] = -1
# catagory[2] = -1
# catagory[7] = -1
# catagory[10] = -1
# catagory[13] = -1
# catagory[14] = -1
weight_type = ['inverse_distance', 'no_weight']

KK = KNNClassifier(catagory, k=3, weight_type=weight_type[0], normalize=False)

KK.fit(all_data[:, :-1], all_data[:, -1])

print('fitted')

filepath = 'Magic_test.arff'
af.load_arff(filepath)
test = af.data
print(KK.score(test[:, :-1], test[:, -1]))
