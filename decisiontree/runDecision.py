import numpy as np
from arff import Arff
from decisiontree import DTClassifier
from sklearn.linear_model import Perceptron

filepath = "test.arff"  # example
af = Arff()
af.load_arff(filepath)
all_data = af.data
dt = DTClassifier()
dt.fit(all_data[:, :-1], all_data[:, -1])
