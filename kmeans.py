import numpy as np
import itertools 
import pdb
import tqdm
from sklearn.cluster import KMeans
def Set_data_kmeans( input, n_clusters):
    target_data = []
    
    for batch in input:
        target_data=np.append(target_data, batch[:,-1:].cpu())
    model = KMeans(n_clusters,random_state=42)
    
    model.fit(target_data[:,np.newaxis])
    return model.cluster_centers_