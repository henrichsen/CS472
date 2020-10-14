import numpy as np
from arff import Arff
from mlp import MLPClassifier
from sklearn.linear_model import Perceptron

def hot(y):
    y=y.flatten()
    values =np.unique(y)
    hoty =np.zeros((len(y),len(values)), dtype=int)
    for i in range(len(values)):
        hoty[y==values[i],i]=1
    return hoty


#filepath = "linsep.arff" #test set
filepath = "data_banknote.arff" #graded set
#filepath = 'iris.arff'
#filepath = 'vowel.arff'
af = Arff()
af.load_arff(filepath)

alldata = af.data
#alldata = alldata[:,3:] # for vowel data set
alllength = len(alldata)
shuffle = False
spliting = False
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
#y = hot(y)
mlp = MLPClassifier(lr=.1, momentum=0.5
                    , shuffle=shuffle, hidden_layer_widths=None)
#print(y)
# X=[[0,0],[0,1]]
# y=[[1],[0]]
#initial_weights={'hidden':[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]] ,'output':[[0,0,0,0,0,0,0,0,0]]}
hl, ol = mlp.fit(X, y,)
if split:
    X1 = testingset[:, :y_start]
    y1 = testingset[:, y_start:]
    score = mlp.score(X1, hot(y1), hl, ol)
    print("The Training set accuracy is: "+str(score))

# print(alldata)

