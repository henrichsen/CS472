from Kmeans import KMEANSClustering
import numpy as np
from arff import Arff

filepath = 'abalone.arff'
af = Arff()
af.load_arff(filepath)

alldata = af.data
split = int(.75 * len(alldata))
shuffle = False
if shuffle:
    np.random.shuffle(alldata)
trainingdata = alldata[:split]
testdata = alldata[split:]
kmean = KMEANSClustering(debug=False)
kmean.fit(testdata)
