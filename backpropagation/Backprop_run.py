import numpy as np
from arff import Arff
from mlp import MLPClassifier
from sklearn.linear_model import Perceptron

filepath = "linsep.arff" #test set
#filepath = "data_banknote.arff" #graded set
af = Arff()
af.load_arff(filepath)

alldata = af.data
alllength = len(alldata)
shuffle = True
spliting = True
y_start = 2
if shuffle:
    np.random.shuffle(alldata)
if spliting:
    split = int(round(.75*alllength))
    print(split)
    trainingset=alldata[:split]
    validset=alldata[split:]
else:
    trainingset=alldata
    validset=alldata
X = trainingset[:,:y_start]
y = trainingset[:,y_start:]

mlp = MLPClassifier(lr=.1, momentum=0, shuffle=shuffle)
mlp.fit(X,y)


print(alldata)
