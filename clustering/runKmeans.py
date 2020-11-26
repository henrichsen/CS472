from Kmeans import KMEANSClustering
import numpy as np
from arff import Arff

filepath = 'iris.arff'
af = Arff()
af.load_arff(filepath)

alldata = af.data
split = int(.75 * len(alldata))
shuffle = False
if shuffle:
    np.random.shuffle(alldata)
trainingdata = alldata[:split]
testdata = alldata[:,:]
for i in range(5):
    print('run= '+str(i))
    kmean = KMEANSClustering(k=4, debug=False)
    kmean.fit(testdata)
