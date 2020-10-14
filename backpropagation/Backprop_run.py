import numpy as np
from arff import Arff
from mlp import MLPClassifier
from sklearn.linear_model import Perceptron

# filepath = "linsep.arff" #test set
# filepath = "data_banknote.arff" #graded set
filepath = 'iris.arff'
af = Arff()
af.load_arff(filepath)

alldata = af.data
alllength = len(alldata)
shuffle = True
spliting = True
y_start = 4
if shuffle:
    np.random.shuffle(alldata)
if spliting:
    split = int(round(.75 * alllength))
    print(split)
    trainingset = alldata[:split]
    testingset = alldata[split:]
else:
    trainingset = alldata
    testingset = alldata
X = trainingset[:, :y_start]
y = trainingset[:, y_start:]

mlp = MLPClassifier(lr=.1, momentum=0, shuffle=shuffle, hidden_layer_widths=None)
# X=[[0,0],[0,1]]
# y=[[1],[0]]
# initial_weights={'hidden':[[1,1,1],[1,1,1]] ,'output':[[1,1,1]]}
hl, ol = mlp.fit(X, y, )
if split:
    X1 = testingset[:, :y_start]
    y1 = testingset[:, y_start:]
    score = mlp.score(X1, y1, hl, ol)
    print("The Training set accuracy is: "+str(score))

# print(alldata)
