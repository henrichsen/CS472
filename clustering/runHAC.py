from HAC import HACClustering
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
testdata = alldata
hac = HACClustering(k=2, link_type='complete', print_k=[1, 2, 3, 4, 5])
hac.fit(testdata)
