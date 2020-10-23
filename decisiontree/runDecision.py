import numpy as np
from arff import Arff
from decisiontree import DTClassifier
from sklearn.linear_model import Perceptron

filepath = "linsep2nonorigin.arff" #example
af = Arff()
af.load_arff(filepath)
