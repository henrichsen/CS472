import numpy as np
from arff import Arff
from KNN import KNNClassifier

filepath = 'seismic-bumps_train.arff'
af = Arff()
af.load_arff(filepath)
all_data = af.data
catagory= np.zeros([len(all_data[0])])

KK=KNNClassifier(catagory)
KK.fit(all_data[:,:-1],all_data[:,-1])
filepath = 'seismic-bumps_test.arff'
af.load_arff(filepath)
test = af.data
KK.predict(test[:,:-1])
