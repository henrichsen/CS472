from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans as kmeans
from sklearn.cluster import AgglomerativeClustering as HAC
import numpy as np
from arff import Arff
from sklearn.metrics import silhouette_score as Score
from sklearn.metrics import davies_bouldin_score as BDScore

filepath = 'iris.arff'
af = Arff()
af.load_arff(filepath)
alldata = af.data
# for i in range(8):
#     if i < 2:
#         continue
#     km = kmeans(n_clusters=i)
#     km.fit(alldata)
#     print(BDScore(alldata, km.predict(alldata)))
print('Single')
distance=['euclidean','l1','l2','manhattan','cosine']
for i in range(len(distance)):
    # if i<2:
    #     continue
    hc=HAC(n_clusters=5,linkage='single',affinity=distance[i])
    hc.fit(alldata)
    print(Score(alldata, hc.fit_predict(alldata)))
# print('complete')
# for i in range(8):
#     if i<2:
#         continue
#     hc=HAC(n_clusters=i,linkage='complete')
#     hc.fit(alldata)
#     print(Score(alldata, hc.fit_predict(alldata)))